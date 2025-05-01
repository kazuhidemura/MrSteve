import copy
from dataclasses import dataclass
from typing import Optional

import gym
import numpy as np

from minedojo.sim import MineDojoSim
from minedojo.sim.wrappers import FastResetWrapper
from minedojo.tasks import _parse_inventory_dict
from minedojo.tasks.meta.utils import (
    check_success_base,
    reward_fn_base,
    simple_inventory_based_check,
    simple_inventory_based_reward,
)


@dataclass
class Task:
    target_item: str
    success_criteria: check_success_base
    reward_fn: reward_fn_base
    initial_inventory: dict


class SequentialTask(gym.Wrapper):
    def __init__(
        self,
        *,
        initial_mobs = None,
        initial_mob_spawn_range_low = None,
        initial_mob_spawn_range_high = None,
        fast_reset: bool = True,
        fast_reset_random_teleport_range: Optional[int] = None,
        fast_reset_random_teleport_range_high: Optional[int] = None,
        fast_reset_random_teleport_range_low: Optional[int] = None,
        tasks,
        **kwargs,
    ):
        sim = MineDojoSim(**kwargs)
        self._fast_reset = fast_reset
        if fast_reset:
            sim = FastResetWrapper(
                sim,
                random_teleport_range=fast_reset_random_teleport_range,
                random_teleport_range_high=fast_reset_random_teleport_range_high,
                random_teleport_range_low=fast_reset_random_teleport_range_low,
            )
        super().__init__(env=sim)

        self._ini_info_dict = None
        self._pre_info_dict = None
        self._elapsed_timesteps = None
        self._success_tasks = 0
        self._total_tasks = len(tasks)

        self._tasks = []
        for task in tasks:
            task = task.copy()
            target_item = task["target_item"]
            success_criteria = simple_inventory_based_check(target_item, 1)
            reward_fn = simple_inventory_based_reward(target_item, 1)
            initial_inventory = task.pop("initial_inventory", {})
            self._tasks.append(
                Task(
                    target_item=target_item,
                    success_criteria=success_criteria,
                    reward_fn=reward_fn,
                    initial_inventory=initial_inventory,
                )
            )

        initial_mobs = initial_mobs or []
        if isinstance(initial_mobs, str):
            initial_mobs = [initial_mobs]
        elif isinstance(initial_mobs, dict):
            initial_mobs = [initial_mobs["name"] for _ in range(initial_mobs["quantity"])]
        elif isinstance(initial_mobs, list) and len(initial_mobs) > 0 and isinstance(initial_mobs[0], dict):
            _initial_mobs = []
            for mob in initial_mobs:
                _initial_mobs.extend([mob["name"] for _ in range(mob["quantity"])])
            initial_mobs = _initial_mobs
        self._initial_mobs = initial_mobs
        if len(initial_mobs) > 0:
            assert len(initial_mob_spawn_range_low) == 3
            assert len(initial_mob_spawn_range_high) == 3
            low = np.repeat(
                np.array(initial_mob_spawn_range_low)[np.newaxis, ...],
                len(initial_mobs),
                axis=0,
            )
            high = np.repeat(
                np.array(initial_mob_spawn_range_high)[np.newaxis, ...],
                len(initial_mobs),
                axis=0,
            )
            self._mob_spawn_range_space = gym.spaces.Box(
                low=low, high=high
            )

    @property
    def success_tasks(self) -> int:
        return self._success_tasks

    @property
    def total_tasks(self) -> int:
        return self._total_tasks

    @property
    def current_target(self) -> Optional[str]:
        target_item = None
        if self._success_tasks < self._total_tasks:
            target_item = self._tasks[self._success_tasks].target_item
        return target_item

    def reset(self, move_flag = True):
        self._elapsed_timesteps = 0
        self._success_tasks = 0

        if self._fast_reset:
            obs = self.env.reset(move_flag=move_flag)
        else:
            obs = self.env.reset()

        self.env.clear_inventory()
        obs, _, _, info = self.env.set_inventory(
            _parse_inventory_dict(self._tasks[self._success_tasks].initial_inventory)
        )
        obs, info = self._after_sim_reset_hook(obs, info)
        self._process_info(info)
        
        self._ini_info_dict = (
            self.env.info_prev_reset or info if self._fast_reset else info
        )
        self._pre_info_dict = copy.deepcopy(info)

        self._process_obs(obs)
        return obs

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        self._elapsed_timesteps += 1

        if self._success_tasks < self._total_tasks:
            cur_task = self._tasks[self._success_tasks]
            reward = cur_task.reward_fn(
                ini_info_dict=self._ini_info_dict,
                pre_info_dict=self._pre_info_dict,
                cur_info_dict=info,
                elapsed_timesteps=self._elapsed_timesteps,
            )
            success = cur_task.success_criteria(
                ini_info_dict=self._ini_info_dict,
                cur_info_dict=info,
                elapsed_timesteps=self._elapsed_timesteps,
            )
            if success:
                self._success_tasks += 1
                if self._success_tasks < self._total_tasks:
                    self.env.clear_inventory()
                    obs, _, _, info = self.env.set_inventory(
                        _parse_inventory_dict(self._tasks[self._success_tasks].initial_inventory)
                    )
        else:
            reward = 0

        done = self.env.is_terminated or (self._success_tasks >= self._total_tasks)
        self._process_info(info)
        self._pre_info_dict = copy.deepcopy(info)
        self._process_obs(obs)
        return obs, reward, done, info

    def _process_obs(self, obs):
        target_item = None
        if self._success_tasks < self._total_tasks:
            target_item = self._tasks[self._success_tasks].target_item
        obs["task"] = target_item

    def _process_info(self, info):
        info["success_tasks"] = self._success_tasks
        info["total_tasks"] = self._total_tasks

        target_item = "none"
        if self._success_tasks < self._total_tasks:
            target_item = self._tasks[self._success_tasks].target_item
        info["task"] = target_item

    def _after_sim_reset_hook(
        self, reset_obs, reset_info
    ):
        obs, info = reset_obs, reset_info

        # to remove black screen at the very beginning
        for _ in range(200):
            obs, _, _, info = self.env.step(self.env.action_space.no_op())

        if len(self._initial_mobs) > 0:
            mobs_rel_positions = self._mob_spawn_range_space.sample()
            obs, _, _, info = self.env.spawn_mobs(
                self._initial_mobs, mobs_rel_positions
            )
        return obs, info
