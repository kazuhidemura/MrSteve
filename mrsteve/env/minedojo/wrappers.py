import cv2
import json
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
        self.video_mode_path = os.path.join(logdir, "video_mode")
        os.makedirs(self.video_mode_path, exist_ok=True)

        video_shape = self.env.observation_space.spaces["rgb"].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(os.path.join(self.video_path, f"tracking_{self.eval_info.episode_id:03d}.mp4"),
                                            fourcc, 20, (video_shape[2], video_shape[1]))
        self.video_mode_writer = cv2.VideoWriter(
            os.path.join(self.video_mode_path, f"tracking_{self.eval_info.episode_id:03d}.mp4"),
            fourcc,
            20,
            (video_shape[2], video_shape[1]),
        )
        self._agent_mode_text = "INVALID"
        self._mode_overlay_padding = 12
        self._mode_overlay_font = cv2.FONT_HERSHEY_SIMPLEX
        self._mode_overlay_font_scale = 0.6
        self._mode_overlay_thickness = 2
        self._mode_overlay_bg_padding = 6

    def set_agent_mode(self, mode_text: str):
        self._agent_mode_text = (mode_text or "UNKNOWN")

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
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

        if self.video_mode_writer is not None:
            overlay_frame = frame_bgr.copy()
            text = f"Agent Mode: {self._agent_mode_text}"
            text_size, baseline = cv2.getTextSize(
                text,
                self._mode_overlay_font,
                self._mode_overlay_font_scale,
                self._mode_overlay_thickness,
            )
            padding = self._mode_overlay_padding
            bg_pad = self._mode_overlay_bg_padding
            x = max(0, overlay_frame.shape[1] - text_size[0] - padding)
            y = max(text_size[1] + padding, text_size[1])
            top_left = (
                max(x - bg_pad, 0),
                max(y - text_size[1] - bg_pad, 0),
            )
            bottom_right = (
                min(overlay_frame.shape[1], x + text_size[0] + bg_pad),
                min(overlay_frame.shape[0], y + baseline + bg_pad),
            )
            cv2.rectangle(overlay_frame, top_left, bottom_right, (0, 0, 0), -1)
            cv2.putText(
                overlay_frame,
                text,
                (x, y),
                self._mode_overlay_font,
                self._mode_overlay_font_scale,
                (255, 255, 255),
                self._mode_overlay_thickness,
                cv2.LINE_AA,
            )
            self.video_mode_writer.write(overlay_frame)

        self.log["task"].append(obs["task"])

    def _save_tracking(self):
        self.video_writer.release()
        if self.video_mode_writer is not None:
            self.video_mode_writer.release()
        np.savez_compressed(os.path.join(self.episode_path, f"tracking_{self.eval_info.episode_id:03d}.npz"), **self.log)

        location_entries = []
        for step, location in enumerate(self.log.get("location", [])):
            entry = {
                "step": step,
                "x": float(location[0]),
                "y": float(location[1]),
                "z": float(location[2]),
                "pitch": float(location[3]),
                "yaw": float(location[4]),
            }
            if len(location) > 5:
                entry["target_x"] = float(location[5])
                entry["target_z"] = float(location[6])
            location_entries.append(entry)

        location_payload = {
            "episode_id": int(self.eval_info.episode_id),
            "locations": location_entries,
        }

        json_path = os.path.join(self.logdir, f"agent_location_{self.eval_info.episode_id:03d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(location_payload, f, indent=2)
