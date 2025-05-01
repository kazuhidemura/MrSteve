import cv2
import os
import numpy as np

from mrsteve.env.common.types import EvalInfo
from mrsteve.env.common.wrappers import BaseEvalTrackingWrapper


class EvalTrackingWrapper(BaseEvalTrackingWrapper):
    def __init__(self, env, logdir, eval_info: EvalInfo, **kwargs):
        super(EvalTrackingWrapper, self).__init__(env, logdir, eval_info)

        self.log["location"] = []
        self.log["success_tasks"] = 0
        self.log["total_tasks"] = 0
        self.log["task_id"] = []
        self.log["task"] = []

        self.video_path = os.path.join(logdir, "video")
        os.makedirs(self.video_path, exist_ok=True)

        video_shape = self.env.observation_space.spaces["rgb"].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(os.path.join(self.video_path, f"tracking_{self.eval_info.episode_id:03d}.mp4"),
                                            fourcc, 20, (video_shape[2], video_shape[1]))

    def _update_from_info(self, info):
        xpos = info["xpos"]
        ypos = info["ypos"]
        zpos = info["zpos"]
        pitch = info["pitch"]
        yaw = info["yaw"]

        position = (xpos, ypos, zpos, pitch, yaw)
        if "target_pos" in info:
            target_xpos, target_zpos = info["target_pos"]
            position += (target_xpos, target_zpos)
        self.log["location"].append(position)

        if "success_tasks" in info:
            self.log["success_tasks"] = info["success_tasks"]
            self.log["task_id"].append(info["success_tasks"])
            self.log["total_tasks"] = info["total_tasks"]

    def _update_after_reset(self):
        xpos = self.env.prev_info["xpos"]
        ypos = self.env.prev_info["ypos"]
        zpos = self.env.prev_info["zpos"]
        pitch = self.env.prev_info["pitch"]
        yaw = self.env.prev_info["yaw"]

        position = (xpos, ypos, zpos, pitch, yaw)
        if "target_pos" in self.env.prev_info:
            target_xpos, target_zpos = self.env.prev_info["target_pos"]
            position += (target_xpos, target_zpos)
        self.log["location"].append(position)

        if "total_tasks" in self.env.prev_info:
            self.log["total_tasks"] = self.env.prev_info["total_tasks"]
        self.log["task_id"].append(0)

    def _update_from_obs(self, obs):
        frame = obs["rgb"].transpose(1, 2, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

        self.log["task"].append(obs["task"])

    def _save_tracking(self):
        self.video_writer.release()
        np.savez_compressed(os.path.join(self.episode_path, f"tracking_{self.eval_info.episode_id:03d}.npz"), **self.log)
