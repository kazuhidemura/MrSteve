import torch as th
import numpy as np
from minedojo.sim.wrappers.fast_reset import FastResetWrapper

from ..base import (
    ClipReward,
    RewardWrapper,
    SuccessWrapper,
    TerminalWrapper,
    VPTWrapper,
)


def name_match(target_name, obs_name):
    return target_name.replace(" ", "_") == obs_name.replace(" ", "_")


# Fast reset wrapper saves time but doesn't replace blocks
# Occasionally doing a hard reset should prevent state shift
class MinedojoSemifastResetWrapper(FastResetWrapper):

    def __init__(
        self,
        *args,
        reset_freq=100,
        random_teleport_range=200,
        apply_start_position_on_reset=False,
        fixed_start_position=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            random_teleport_range=random_teleport_range,
            random_teleport_range_high=random_teleport_range,
            random_teleport_range_low=0,
            apply_start_position_on_reset=apply_start_position_on_reset,
            **kwargs,
        )
        self.reset_freq = reset_freq
        self.reset_count = 0

        if fixed_start_position is not None:
            self._start_position = dict(fixed_start_position)
            self.init_position = {
                "x": self._start_position["x"],
                "y": self._start_position["y"],
                "z": self._start_position["z"],
            }

    def reset(self):
        if self.reset_count < self.reset_freq:
            self.reset_count += 1
            obs = super().reset()
        else:
            self.reset_count = 0
            obs = self.env.reset()

        if self._start_position is not None and not self._apply_start_position_on_reset:
            obs, _, _, _ = self.teleport_agent(**self._start_position)
            self.init_position = {
                "x": self._start_position["x"],
                "y": self._start_position["y"],
                "z": self._start_position["z"],
            }
            self.birth_position = {
                "x": self._start_position["x"],
                "y": self._start_position["y"],
                "z": self._start_position["z"],
            }

        return obs


class MinedojoClipReward(ClipReward):
    @staticmethod
    def _get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        return th.from_numpy(curr_frame)

    @staticmethod
    def get_resolution():
        return (160, 256)

class MinedojoRewardWrapper(RewardWrapper):
    @staticmethod
    def _get_item_count(obs, item):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) if name_match(item, name))


class MinedojoSuccessWrapper(SuccessWrapper):
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False


class MinedojoTerminalWrapper(TerminalWrapper):
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False

    @staticmethod
    def _check_death_condition(condition_info, obs):
        return obs["life_stats"]["life"].item() == 0


class MinedojoVPTWrapper(VPTWrapper):
    def _filter_actions(self, actions):
        filtered_actions = {
            handler.to_string(): actions[handler.to_string()] 
                if handler.to_string() in actions else handler.space.no_op()
            for handler in self.unwrapped._sim_spec.actionables  # This comes from MinedojoSim.SimSpec
        }  # Filter malmo actions by what current minedojo task has enabled
        return filtered_actions

    def _process_obs(self, obs):
        obs["rgb"] = np.transpose(obs["rgb"], (1, 2, 0))
        return obs
