import numpy as np
from typing import Dict
from minedojo.sim import MineDojoSim

from mrsteve.env.minedojo.task.base.info_wrapper import InfoWrapper

from ..minedojo.wrappers import (
    MinedojoTerminalWrapper,
    MinedojoSemifastResetWrapper,
)
from .task_base import SequentialTask

FIXED_WORLD_SEED = 1871
SEED_OFFSET_BY_EPISODE = False
FIXED_TELEPORT_POSITION = dict(x=-229, y=67, z=250, yaw=-126.7, pitch=7.4)


_MetaTaskName2Class = {
    "Sequential": SequentialTask,
}
MetaTaskName2Class = {k.lower(): v for k, v in _MetaTaskName2Class.items()}


def _get_minedojo_specs(episode_id, task_id, task_specs, sim_specs):
    minedojo_specs = dict()
    meta_task_cls = task_id

    minedojo_specs.update(dict(
        image_size=(160, 256),
        fast_reset=False,
        event_level_control=True,
        use_lidar=False,
        use_voxel=True,
        voxel_size=dict(xmin=-1, ymin=0, zmin=1, xmax=1, ymax=1, zmax=2),
    ))

    minedojo_specs.update(**sim_specs)

    base_seed = FIXED_WORLD_SEED
    seed = base_seed + episode_id if SEED_OFFSET_BY_EPISODE else base_seed
    if minedojo_specs.get("generate_world_type") == "specified_biome":
        minedojo_specs["seed"] = seed
        minedojo_specs["world_seed"] = seed

    minedojo_specs["start_position"] = FIXED_TELEPORT_POSITION.copy()

    return meta_task_cls, minedojo_specs


def _add_wrappers(
    env,
    task_id: str,
    terminal_specs: Dict = None,
    fast_reset: int = None,
    **kwargs,
):
    if terminal_specs is None:
        terminal_specs = dict(max_steps=500, on_death=True)
    env = MinedojoTerminalWrapper(env, **terminal_specs)
    env = InfoWrapper(env)

    if fast_reset is not None:
        wrapped = env
        while hasattr(wrapped, "env"):
            if isinstance(wrapped.env, MineDojoSim):
                wrapped.env = MinedojoSemifastResetWrapper(
                    wrapped.env,
                    reset_freq=fast_reset,
                    random_teleport_range=200,
                    fixed_start_position=FIXED_TELEPORT_POSITION.copy(),
                )
                break
            wrapped = wrapped.env

    return env


def make_minedojo(episode_id: int, task_id: str, task_specs, sim_specs):
    meta_task_cls, minedojo_specs = _get_minedojo_specs(episode_id, task_id, task_specs, sim_specs)

    meta_task = meta_task_cls.lower()
    assert (
        meta_task in MetaTaskName2Class
    ), f"Invalid meta task name provided: {meta_task}"

    env = MetaTaskName2Class[meta_task](**minedojo_specs)
    env = _add_wrappers(env, task_id, **task_specs)

    return env
