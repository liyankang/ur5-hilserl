#!/usr/bin/env python3

import sys
from pathlib import Path

# 直接运行脚本时也能找到项目根目录下的公共采集工具。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import jax
import jax.numpy as jnp
from natsort import natsorted
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import glob
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from pynput import keyboard

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.timer_utils import Timer

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from examples.collection_utils import (
    apply_manual_success, ensure_training_schema, make_transition,
    validate_action_space, validate_parameter_shapes, validate_transition_shapes,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("demo_buffer_path", None, "Path to folder of demo buffers.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("pretrain_steps", 20_000, "Number of pretraining steps.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

should_reset = False

def on_press(key):
    global should_reset
    if key == keyboard.Key.esc:
        should_reset = True
        print("ESC pressed. Resetting...")

# Start the keyboard listener in a non-blocking way


##############################################################################


def actor(agent: BCAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        if not FLAGS.checkpoint_path or not os.path.exists(os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.eval_checkpoint_step}")):
            raise FileNotFoundError(f"Requested BC checkpoint_{FLAGS.eval_checkpoint_step} does not exist.")
        global should_reset
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        validate_parameter_shapes(agent.state.params, ckpt.params, "BC evaluation checkpoint")
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                done = bool(done or truncated)

                if done or should_reset:
                    should_reset = False
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")
                    break

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer"))
        else 0
    )


    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        validate_parameter_shapes(agent.state.params, params, "BC learner parameter update")
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=False,
            )
            actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            reward, done, truncated, info = apply_manual_success(reward, done, truncated, info, env=env)
            if "left" in info:
                info.pop('left')
            if "right" in info:
                info.pop('right')

            # override the action with the intervention action
            if info.get("is_intervention", "intervene_action" in info):
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            info["is_intervention"] = already_intervened
            actions = info.get("applied_action", info.get("intervene_action", actions))

            reward = np.asarray(reward, dtype=np.float32)
            running_return += reward
            transition = make_transition(obs, actions, next_obs, reward, done, truncated, info)
            if already_intervened:
                data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info.setdefault("episode", {})
                info['episode']['intervention_count'] = intervention_count
                info['episode']['intervention_steps'] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: BCAgent, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", demo_buffer)
    server.start(threaded=True)

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # Pretrain BC policy to get started
    update_step = 0
    if FLAGS.pretrain_steps:
        if os.path.isdir(
            os.path.join(
                FLAGS.checkpoint_path, f"checkpoint_{FLAGS.pretrain_steps}"
            )
        ):
            print_green(
                f"BC checkpoint at {FLAGS.pretrain_steps} steps found, restoring BC checkpoint"
            )
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=FLAGS.pretrain_steps
            )
            validate_parameter_shapes(agent.state.params, ckpt.params, "BC training checkpoint")
            agent = agent.replace(state=ckpt)
            update_step = FLAGS.pretrain_steps
        else:
            update_step = 0
            print_yellow(
                f"No BC checkpoint at {FLAGS.pretrain_steps} steps found, starting from scratch"
            )
            for step in tqdm.tqdm(
                range(FLAGS.pretrain_steps),
                dynamic_ncols=True,
                desc="bc_pretraining",
            ):
                update_step += 1
                batch = next(demo_iterator)
                agent, bc_update_info = agent.update(batch)
                if update_step % config.log_period == 0 and wandb_logger:
                    wandb_logger.log({"bc": bc_update_info}, step=update_step)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=update_step, keep=20
            )
            print_green("bc pretraining done and saved checkpoint")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.pretrain_steps+1, config.max_steps), dynamic_ncols=True, desc="learner"):

        with timer.context("train"):
            batch = next(demo_iterator)
            agent, update_info = agent.update(
                batch,
            )
  
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )



##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # CLI 显式值优先；缺省时使用任务 JSON 中的模型保存目录。
    if FLAGS.checkpoint_path is None and getattr(config, "checkpoint_path", None):
        FLAGS.checkpoint_path = config.checkpoint_path

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    env = config.get_environment(fake_env=FLAGS.learner, 
                                 save_video=False, 
                                 classifier=FLAGS.actor)
    env = RecordEpisodeStatistics(env)
    sample_action = validate_action_space(config, env)
    latest_checkpoint = (
        checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path) else None
    )
    ensure_training_schema(config, env, FLAGS.checkpoint_path, FLAGS.exp_name,
                           has_checkpoint=latest_checkpoint is not None, algorithm="bc")

    rng, sampling_rng = jax.random.split(rng)
    agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=sample_action,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: BCAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=50000,
            image_keys=config.image_keys,
            include_intervention=getattr(config, "action_dim", None) is not None,
        )

        assert FLAGS.demo_path is not None or FLAGS.demo_buffer_path is not None

        for file in glob.glob(os.path.join(FLAGS.demo_buffer_path, "*.pkl")) if FLAGS.demo_buffer_path else []:
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    validate_transition_shapes(transition, env.observation_space, env.action_space, file)
                    transition.setdefault("is_intervention", True)
                    demo_buffer.insert(transition)
        print_green(
            f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
        )
        for path in FLAGS.demo_path or []:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    validate_transition_shapes(transition, env.observation_space, env.action_space, path)
                    transition.setdefault("is_intervention", True)
                    demo_buffer.insert(transition)
        print(f"demo buffer size: {len(demo_buffer)}")
        
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        
        actor(
            agent, 
            data_store,
            env, 
            sampling_rng, 
            )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
