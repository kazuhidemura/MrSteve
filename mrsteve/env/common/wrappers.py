from gym import Wrapper
import numpy as np
import pickle
import os

from mrsteve.env.common.types import EvalInfo


class BaseEvalTrackingWrapper(Wrapper):
    def __init__(self, env, logdir, eval_info: EvalInfo, **kwargs):
        super(BaseEvalTrackingWrapper, self).__init__(env)

        self.eval_info = eval_info

        self.logdir = logdir
        self.obs_log = []
        self.log = dict(
            reward=[],
        )

        self.episode_path = os.path.join(logdir, "episode")
        os.makedirs(self.episode_path, exist_ok=True)

    def close(self):
        self._save_tracking()
        super().close()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._update_from_obs(obs)
        self._update_after_reset()

        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)

        self._update_from_obs(obs)
        self.log["reward"].append(reward)
        self._update_from_info(info)

        return obs, reward, done, info

    def _update_after_reset(self):
        pass

    def _update_from_info(self, info):
        pass

    def _update_from_obs(self, obs):
        self.obs_log.append(obs)

    def _save_tracking(self):
        self.log["observation"] = self.obs_log
        np.savez_compressed(os.path.join(self.episode_path, f"tracking_{self.eval_info.episode_id:03d}.npz"), **self.log)


class ActionTrackingWrapper(Wrapper):
    def __init__(self, env, logdir, eval_info: EvalInfo, **kwargs):
        super(ActionTrackingWrapper, self).__init__(env)

        self.eval_info = eval_info

        self.logdir = logdir
        self.actions = []

        self.action_path = os.path.join(logdir, "action")
        os.makedirs(self.action_path, exist_ok=True)

    def __del__(self):
        filepath = os.path.join(self.action_path, f"action_tracking_{self.eval_info.episode_id:03d}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self.actions, f)

    def step(self, action, **kwargs):
        step_return = self.env.step(action, **kwargs)
        self.actions.append(action)
        return step_return
