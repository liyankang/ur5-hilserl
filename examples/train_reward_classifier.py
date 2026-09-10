import glob
import os
import sys
import pickle as pkl
import gymnasium as gym
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "ram_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("seed", 0, "Random seed for train/validation split.")
flags.DEFINE_float("val_ratio", 0.2, "Fraction of each class to hold out for validation.")
flags.DEFINE_integer("eval_period", 10, "Evaluate validation metrics every N epochs.")
flags.DEFINE_boolean(
    "terminal_only",
    True,
    "Train the classifier on final/terminal observations so it learns successful end states.",
)


def _normalize_classifier_obs(obs, classifier_keys):
    if "images" not in obs:
        return None

    key_map = {"wrist_1": "global_1", "wrist_2": "wrist", "wrist_3": "global_2"}
    new_obs = {"state": obs["state"]}
    for ck in classifier_keys:
        if ck in obs["images"]:
            new_obs[ck] = obs["images"][ck]
            continue
        for old_k, new_k in key_map.items():
            if new_k == ck and old_k in obs["images"]:
                new_obs[ck] = obs["images"][old_k]
                break
    if any(ck not in new_obs for ck in classifier_keys):
        return None
    return new_obs


def _transition_to_classifier_example(trans, label, classifier_keys):
    obs_key = "next_observations" if "next_observations" in trans else "observations"
    obs = _normalize_classifier_obs(trans[obs_key], classifier_keys)
    if obs is None:
        return None

    example = dict(trans)
    example["observations"] = obs
    example["next_observations"] = obs
    example["labels"] = label
    return example


def _split_examples(examples, val_ratio, rng):
    if len(examples) < 2:
        return examples, []
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(examples) * val_ratio)))
    val_count = min(val_count, len(examples) - 1)
    val_indices = set(indices[:val_count])
    train = [example for i, example in enumerate(examples) if i not in val_indices]
    val = [example for i, example in enumerate(examples) if i in val_indices]
    return train, val


def _make_buffer(examples, observation_space, action_space, capacity):
    buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=max(capacity, len(examples), 1),
        include_label=True,
    )
    for example in examples:
        buffer.insert(example)
    return buffer


def _format_batch(batch):
    return batch.copy(
        add_or_replace={
            "labels": batch["labels"][..., None],
        }
    )


def _score_summary(name, probs):
    if probs.size == 0:
        return f"{name}: empty"
    quantiles = np.percentile(probs, [0, 10, 25, 50, 75, 90, 100])
    return (
        f"{name}: n={probs.size}, mean={probs.mean():.4f}, "
        f"p0={quantiles[0]:.4f}, p10={quantiles[1]:.4f}, p25={quantiles[2]:.4f}, "
        f"p50={quantiles[3]:.4f}, p75={quantiles[4]:.4f}, "
        f"p90={quantiles[5]:.4f}, p100={quantiles[6]:.4f}"
    )


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Only keep classifier-relevant keys in observation space (flat structure)
    kept_keys = ["state"] + list(config.classifier_keys)
    filtered_obs_space = gym.spaces.Dict(
        {k: v for k, v in env.observation_space.spaces.items() if k in kept_keys}
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    pos_examples = []
    num_skipped_success = 0
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if FLAGS.terminal_only and not trans.get("dones", False):
                continue
            example = _transition_to_classifier_example(trans, 1, config.classifier_keys)
            if example is None:
                num_skipped_success += 1
                continue
            example["actions"] = env.action_space.sample()
            pos_examples.append(example)
            
    if len(pos_examples) == 0:
        raise ValueError("No positive classifier samples found. Check success data or disable --terminal_only.")
    
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    neg_examples = []
    num_skipped_failure = 0
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if FLAGS.terminal_only and not trans.get("dones", False):
                continue
            example = _transition_to_classifier_example(trans, 0, config.classifier_keys)
            if example is None:
                num_skipped_failure += 1
                continue
            example["actions"] = env.action_space.sample()
            neg_examples.append(example)
            
    if len(neg_examples) == 0:
        raise ValueError("No negative classifier samples found. Check failure data or disable --terminal_only.")

    split_rng = np.random.default_rng(FLAGS.seed)
    pos_train, pos_val = _split_examples(pos_examples, FLAGS.val_ratio, split_rng)
    neg_train, neg_val = _split_examples(neg_examples, FLAGS.val_ratio, split_rng)
    if len(pos_val) == 0 or len(neg_val) == 0:
        raise ValueError(
            "Validation split is empty for at least one class. "
            "Collect more classifier samples or lower --val_ratio."
        )

    pos_buffer = _make_buffer(pos_train, filtered_obs_space, env.action_space, 20000)
    neg_buffer = _make_buffer(neg_train, filtered_obs_space, env.action_space, 50000)
    pos_val_buffer = _make_buffer(pos_val, filtered_obs_space, env.action_space, len(pos_val))
    neg_val_buffer = _make_buffer(neg_val, filtered_obs_space, env.action_space, len(neg_val))

    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(
        f"failed samples: train={len(neg_buffer)}, val={len(neg_val_buffer)}, "
        f"skipped={num_skipped_failure}"
    )
    print(
        f"success samples: train={len(pos_buffer)}, val={len(pos_val_buffer)}, "
        f"skipped={num_skipped_success}"
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False
        )
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        probs = nn.sigmoid(logits)
        accuracy = jnp.mean((probs >= 0.5) == batch["labels"])
        return loss, accuracy, probs

    def evaluate_classifier(epoch):
        pos_val_batch = _format_batch(
            pos_val_buffer.sample(batch_size=len(pos_val_buffer), indx=np.arange(len(pos_val_buffer)))
        )
        neg_val_batch = _format_batch(
            neg_val_buffer.sample(batch_size=len(neg_val_buffer), indx=np.arange(len(neg_val_buffer)))
        )
        val_batch = concat_batches(pos_val_batch, neg_val_batch, axis=0)
        val_loss, val_accuracy, val_probs = eval_step(classifier, val_batch)
        pos_loss, pos_accuracy, pos_probs = eval_step(classifier, pos_val_batch)
        neg_loss, neg_accuracy, neg_probs = eval_step(classifier, neg_val_batch)

        pos_probs_np = np.asarray(jax.device_get(pos_probs)).reshape(-1)
        neg_probs_np = np.asarray(jax.device_get(neg_probs)).reshape(-1)
        print(
            f"Validation epoch {epoch}: loss={float(val_loss):.4f}, "
            f"accuracy={float(val_accuracy):.4f}, "
            f"pos_acc={float(pos_accuracy):.4f}, neg_acc={float(neg_accuracy):.4f}"
        )
        print(_score_summary("  success probs", pos_probs_np))
        print(_score_summary("  failure probs", neg_probs_np))

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = _format_batch(batch.copy(add_or_replace={"observations": obs}))
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

        if (epoch + 1) % FLAGS.eval_period == 0 or epoch + 1 == FLAGS.num_epochs:
            evaluate_classifier(epoch + 1)

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)
