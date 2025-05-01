import math
import gym
import numpy as np
import torch
from dataclasses import dataclass
import time
import time

from mrsteve.agent.controller.steve1.agent import SKILLS
from mrsteve.agent.controller.mrsteve.lib.memory import MEMORY_CLS


MINECLIP_INPUT_FRAMES = 16

@dataclass
class EpisodicMemoryRecord:
    timestep: int
    frame: np.ndarray
    pos: np.ndarray
    cam: np.ndarray


class EpisodicMemoryWrapper(gym.Wrapper):
    def __init__(self, config, env, mineclip, device):
        super().__init__(env)
        self.skip_frames = config.skip_frames
        self.clip_batchsize = config.clip_batchsize
        self.memory_capacity = config.memory_capacity
        self.memory_option = config.memory_option

        self.mineclip = mineclip
        self.device = device

        self.skipped_frame = 0
        self.timestep = 0
        self.img_buffer = []
        self.query_logs = []
        self.buffer: list[EpisodicMemoryRecord] = []
        
        print(f'the agent is using {config.memory_option} with memory capacity {self.memory_capacity}', flush=True)
        if config.memory_option not in MEMORY_CLS:
            raise NotImplementedError(f'invalid memory option {config.memory_option}')

        mem_cls = MEMORY_CLS[config.memory_option]
        mem_cfg = config[config.memory_option]
        self.memory = mem_cls(mem_cfg)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)

        self.skipped_frame = 0
        self.timestep = 0
        self.buffer = []
        self.query_logs = []
        
        init_pos = np.array([obs["location_stats"]["pos"][0], obs["location_stats"]["pos"][2]])
        init_cam = np.array([obs["location_stats"]["pitch"], obs["location_stats"]["yaw"]])
        init_log = {'init_pos': init_pos, 'init_cam': init_cam}
        self.memory.reset(init_log)

        self._add_buffer(obs)
        self._update_cognitive_map()
        
        return obs

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)

        if self.skipped_frame >= self.skip_frames:
            self._add_buffer(obs)
            self.skipped_frame = 0
        else:
            self.skipped_frame += 1

        if len(self.buffer) >= self.clip_batchsize:
            self._update_cognitive_map()

        self.timestep += 1
        return obs, reward, done, info

    def _add_buffer(self, obs):
        img = obs["rgb"].copy()
        frames = img.reshape(1, *img.shape).repeat(MINECLIP_INPUT_FRAMES, axis=0)
        pos = np.array([obs["location_stats"]["pos"][0], obs["location_stats"]["pos"][2]])
        cam = np.array([obs["location_stats"]["pitch"], obs["location_stats"]["yaw"]])
        self.buffer.append(EpisodicMemoryRecord(self.timestep, frames, pos, cam))

    @torch.no_grad()
    def _update_cognitive_map(self):
        inputs = torch.from_numpy(np.stack([record.frame for record in self.buffer], axis=0)).float().to(self.device)
        B = inputs.shape[0]
        embeds = self.mineclip.forward_image_features(
            inputs.view(B * MINECLIP_INPUT_FRAMES, *inputs.shape[2:])
        ).view(B, MINECLIP_INPUT_FRAMES, -1)
        embeds = self.mineclip.forward_video_features(embeds).cpu().numpy()  # 1,512

        for embed, record in zip(embeds, self.buffer):
            self.memory.add(EpisodicMemoryRecord(
                record.timestep, embed, record.pos, record.cam
            ), self.mineclip, self.device)
        self.buffer = []

    def query(self, target_item: str, clip_threshold: float):
        if len(self.memory) == 0:
            return None, None

        current_pos = np.array([self.env.prev_obs["location_stats"]["pos"][0], self.env.prev_obs["location_stats"]["pos"][2]])
        search_prompt = SKILLS[target_item]["explore"]
        text_embeds = self.mineclip.encode_text([search_prompt])
        
        # query from memory
        start_time = time.time()
        candidate, stats = self.memory.query(current_pos, text_embeds, self.mineclip, clip_threshold)
        end_time = time.time()
    
        # query log
        query_log = {
            'query_time': end_time - start_time,
            'timestep': self.timestep,
        }
        query_log.update(self.memory.get_status())
        self.query_logs.append(query_log)

        if candidate is None:
            return None, stats
        
        return (candidate.pos, candidate.cam), stats
