"""Transition conventions shared by data collectors."""
import copy
import os
import pickle
import tempfile
from pathlib import Path


def save_pickle(path, data):
    """Replace a snapshot atomically so interruption cannot corrupt a resume file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_transition(obs, action, next_obs, reward, terminated, truncated, info):
    """Timeout ends the episode but preserves value bootstrapping."""
    return copy.deepcopy(dict(
        observations=obs, actions=info.get("intervene_action", action),
        next_observations=next_obs, rewards=float(reward),
        masks=float(not terminated), dones=bool(terminated or truncated), infos=info,
    ))


def label_success(transition):
    transition.update(rewards=1.0, masks=0.0, dones=True)
    transition.setdefault("infos", {})["succeed"] = True


def classifier_sample(transition, success):
    """A selected frame is a standalone labeled example for terminal filtering."""
    sample = copy.deepcopy(transition)
    sample.update(labels=int(success), dones=True, masks=0.0, rewards=float(success))
    sample.setdefault("infos", {})["succeed"] = bool(success)
    return sample
