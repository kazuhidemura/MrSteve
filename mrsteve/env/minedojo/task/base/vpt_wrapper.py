import copy
from gym import Wrapper
import gym.spaces as spaces
import numpy as np
from abc import ABC, abstractmethod

from mrsteve.lib.VPT.agent import AGENT_RESOLUTION, ACTION_TRANSFORMER_KWARGS, resize_image
from mrsteve.lib.VPT.lib.action_mapping import CameraHierarchicalMapping
from mrsteve.lib.VPT.lib.actions import ActionTransformer


class VPTWrapper(Wrapper, ABC):
    def __init__(self, env, render=False, freeze_equipped=False):
        super().__init__(env)

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        self.observation_space = env.observation_space
        self.observation_space["rgb"] = spaces.Box(0, 255, shape=AGENT_RESOLUTION+(3,))
        self.action_space = spaces.MultiDiscrete([
            space.eltype.n 
            for space in self.action_mapper.get_action_space_update().values()
        ])

        self.do_render = render
        self.freeze_equipped = freeze_equipped

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(copy.deepcopy(obs))
        return obs

    def step(self, action, use_minerl_action=False):
        env_action = self._process_action(action, use_minerl_action)
        #print('action in:', action, 'action_out:', env_action)
        obs, reward, done, info = self.env.step(env_action)
        if self.do_render:
            self.env.render()

        obs = self._process_obs(copy.deepcopy(obs))
        return obs, reward, done, info

    def _process_action(self, action, use_minerl_action=False):
        if use_minerl_action:
            minerl_action_transformed = action 
        else:
            action = {
                "camera": np.expand_dims(action[0], axis=0),
                "buttons": np.expand_dims(action[1], axis=0)
            }
            minerl_action = self.action_mapper.to_factored(action)
            minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        if self.freeze_equipped:
            for name in ["drop", "swap_slot", "pickItem", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", 
                            "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9"]:
                minerl_action_transformed.pop(name, None)
        return self._filter_actions(minerl_action_transformed)

    @abstractmethod
    def _process_obs(self, obs):
        raise NotImplementedError()

    @abstractmethod
    def _filter_actions(self, actions):
        raise NotImplementedError()
