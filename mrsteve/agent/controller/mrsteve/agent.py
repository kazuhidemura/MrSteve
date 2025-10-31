import cv2
import enum
import numpy as np
import json
from patchify import patchify
from scipy.spatial.distance import cdist


from mrsteve.agent.controller.mrsteve.lib.epmem import EpisodicMemoryWrapper
from mrsteve.agent.controller.steve1.agent import Steve1Agent, SKILLS
from mrsteve.agent.controller.mrsteve.lib.vpt_nav import PointConditionalAgent, load_model_parameters
from mrsteve.env.minedojo.wrappers import EvalTrackingWrapper


with open("mrsteve/data/mineclip_thresholds.json", "rt") as f:
    MINECLIP_THRESHOLDS = json.load(f)


class AgentMode(enum.Enum):
    INVALID = -1
    EXPLORE = 0
    NAVIGATE = 1
    EXECUTE = 2
    CAM_ADJUST = 3


class MrSteveAgent:
    def __init__(
        self,
        epmem_env: EpisodicMemoryWrapper,
        config,
        mineclip,
        device
    ):
        self.epmem: EpisodicMemoryWrapper = epmem_env
        self.mineclip = mineclip
        self.device = device

        # Steve-1
        self.steve1 = Steve1Agent(
            config.steve1.model,
            config.steve1.weights,
            config.steve1.prior_weights,
            config.steve1.cond_scale,
            mineclip,
            device 
        )
        self.steve1.reset()

        # VPT-Nav
        nav_policy_kwargs, nav_pi_head_kwargs = load_model_parameters(config.vpt_nav.model)
        self.vpt_nav = PointConditionalAgent(
            device,
            nav_policy_kwargs,
            nav_pi_head_kwargs
        )
        self.vpt_nav.load_weights(config.vpt_nav.weights)
        self.vpt_nav.reset(config.vpt_nav.cond_scale)
        self.vpt_nav.policy.eval()

        # Agent States
        self.mode = AgentMode.INVALID
        self.mode_count = 0

        self.timestep = 0
        self.agent_log = []

        self.cur_task = None
        self.nav_target = None
        self.cam_target = None
        self.steve1_prompt = None

        self.verbose = config.verbose
        self.explore_early_stopping = config.explore.early_stopping

        # For exploration
        self.reached = True
        self.reach_time = 0
        self.reach_logs = []
        self.first_action = True

        # for uncertainty map
        self.umr = config.explore.unc_map_range
        self.num_points = config.explore.num_points
        self.fov_angle = config.explore.fov_angle
        self.fov_radius = config.explore.fov_radius
        self.action_radius = config.explore.action_radius
        self.reach_margin = config.explore.reach_margin
        self.unc_map = np.zeros((self.umr, self.umr, 3)).astype(np.uint8)

        # To avoid struck
        self.prev_pos = None
        self.eval_tracking_wrapper = self._find_eval_tracking_wrapper(epmem_env)
        if self.eval_tracking_wrapper is not None:
            self.eval_tracking_wrapper.set_agent_mode(self.mode.name)

    def get_action(self, obs):
        cur_pos = obs["location_stats"]["pos"]
        cur_pos = np.array([cur_pos[0], cur_pos[2]])

        if self.first_action:
            self.init_pos = cur_pos.copy()
            self.first_action = False

        self.unc_map = self._update_unc_map(self.unc_map, obs)

        # task change
        if self.cur_task != obs["task"]:
            self._task_change_hook(obs)
            if self.verbose:
                print(f"New task: {self.cur_task}")
            self._invoke_epmem_query(obs)

        # timeout for this execution mode. retry
        if ((self.explore_early_stopping and self.mode == AgentMode.EXPLORE) or
            self.mode_count % 600 == 0 and self.mode_count > 0):
            self._invoke_epmem_query(obs)

        if self.mode == AgentMode.EXPLORE:
            action = self._get_action_from_high_countbased_low_vpt_nav(obs)
        elif self.mode == AgentMode.NAVIGATE:
            if np.linalg.norm(cur_pos - self.nav_target) < 3:
                self._mode_switch(AgentMode.CAM_ADJUST, obs)
            else:
                action = self._get_action_from_vpt_nav(obs)

        if self.mode == AgentMode.CAM_ADJUST:
            ANGLE_LIMIT = 5

            target_pitch = self.cam_target[0]
            target_yaw = self.cam_target[1]

            diff_yaw = target_yaw - obs["location_stats"]["yaw"].item()
            if diff_yaw > 180:
                diff_yaw = diff_yaw - 360
            elif diff_yaw < -180:
                diff_yaw = diff_yaw + 360

            yaw_okay = False
            if -ANGLE_LIMIT < diff_yaw < ANGLE_LIMIT:
                yaw_okay = True

            pitch_okay = False
            diff_pitch = target_pitch - obs["location_stats"]["pitch"].item()
            if -ANGLE_LIMIT < diff_pitch < ANGLE_LIMIT:
                pitch_okay = True

            if pitch_okay and yaw_okay:
                self.steve1_prompt = SKILLS[self.cur_task]["execute"]
                self._mode_switch(AgentMode.EXECUTE, obs)
            else:
                action = {"camera": (diff_pitch, diff_yaw)}

        if self.mode == AgentMode.EXECUTE:
            action = self._get_action_from_steve1(obs)

        self.mode_count += 1
        self.timestep += 1
        self.prev_pos = cur_pos.copy()
        return action

    def _recommend_pos(self, unc_map, obs):
        cur_pos = obs["location_stats"]["pos"]
        x, y = cur_pos[0] - self.init_pos[0], cur_pos[2] - self.init_pos[1]
        yaw = obs["location_stats"]["yaw"].item()
        center_x, center_y, ar, arh = unc_map.shape[0] // 2, unc_map.shape[1] // 2, self.action_radius, self.action_radius // 2
        rad = np.deg2rad(-yaw)
        map_x = int(center_x + x)
        map_y = int(center_y + y)
        
        map_patch_x = ((map_x - arh) // ar)
        map_patch_y = ((map_y - arh) // ar)
        map_patch_pos = np.array([[map_patch_x, map_patch_y]])
        
        unc_map_patch = patchify(unc_map[arh:-arh, arh:-arh], (ar, ar, 3), step=ar)
        
        occupy_patch = (unc_map_patch[..., 0] > 128).sum(axis=(2,3,4))
        
        map_unseen_patch_pos = np.transpose((occupy_patch == 0).nonzero())[:, ::-1]
        
        if len(map_unseen_patch_pos) == 0:
            min_patch_pos = np.array(np.unravel_index(occupy_patch.argmin(), occupy_patch.shape))[::-1]
            map_next_pos = (min_patch_pos + 1) * ar
        else:
            dist = cdist(map_unseen_patch_pos, map_patch_pos).flatten()
            map_next_patch_pos_cand = map_unseen_patch_pos[dist == dist.min()]
            map_next_patch_pos = map_next_patch_pos_cand[int(np.random.randint(len(map_next_patch_pos_cand)))]
            map_next_pos = (map_next_patch_pos + 1) * ar
        
        next_pos = map_next_pos - np.array([center_x, center_y]) + self.init_pos
        return next_pos

    def _update_unc_map(self, unc_map, obs):
        
        cur_pos = obs["location_stats"]["pos"]
        x, y = cur_pos[0] - self.init_pos[0], cur_pos[2] - self.init_pos[1]
        yaw = obs["location_stats"]["yaw"].item()
        center_x, center_y = unc_map.shape[0] // 2, unc_map.shape[1] // 2
        rad = np.deg2rad(-yaw)
        map_x = int(center_x + x)
        map_y = int(center_y + y)
        
        # make polygon vertices for sector 
        angles = np.linspace(-self.fov_angle / 2, self.fov_angle / 2, self.num_points)
        angles = np.deg2rad(angles) 
        sector_points = np.zeros((self.num_points + 1, 2))
        sector_points[0] = [0, 0]
    
        for i, angle in enumerate(angles, start=1):
            sector_points[i] = [self.fov_radius * np.sin(angle),
                                self.fov_radius * np.cos(angle)]
        
        # rotate the sector points based on the yaw
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        rotated_points = np.dot(sector_points, rotation_matrix)
        translated_points = rotated_points + np.array([map_x, map_y])
        translated_points = translated_points.astype(np.int32).reshape((-1, 1, 2))
        unc_map[:, :, 1] = 0  # remove previous agent's position
        cv2.fillPoly(unc_map, [translated_points], color=(255, 255, 0), lineType=cv2.LINE_AA)
        return unc_map

    def _get_action_from_high_countbased_low_vpt_nav(self, obs):
        cur_pos = obs["location_stats"]["pos"]
        cur_pos = np.array([cur_pos[0], cur_pos[2]])

        # propose new explore position based on current map config    
        if self.reached:
            self.nav_target = self._recommend_pos(self.unc_map, obs)
            self.reached = False
            self.reach_logs.append({"time_step": self.timestep,
                                    "cur_tar": [float(self.nav_target[0]), float(self.nav_target[1])],
                                    "cur_pos": [float(cur_pos[0]), float(cur_pos[1])],
                                    })
        else:
            self.reach_time += 1
            if (np.linalg.norm(cur_pos - self.nav_target) < self.reach_margin or
                self.reach_time > 2000):
                self.reach_time = 0
                self.reached = True
                self.reach_logs.append({"time_step": self.timestep,
                                        "cur_pos": [float(cur_pos[0]), float(cur_pos[1])],
                                        "reached": self.reached,
                                        })
            
        action = self._get_action_from_vpt_nav(obs)
        return action 

    def _get_action_from_steve1(self, obs):
        minerl_action = self.steve1.get_action(obs["rgb"])
        return minerl_action

    def _get_action_from_vpt_nav(self, obs):
        obs = obs.copy()
        obs["pov"] = obs["rgb"]
        minerl_action = self.vpt_nav.get_action(obs, self.nav_target)
        return minerl_action

    def _task_change_hook(self, obs):
        self.cur_task = obs["task"]
        if self.cur_task in SKILLS:
            self.steve1_prompt = SKILLS[self.cur_task]["execute"]
        self._mode_switch(AgentMode.INVALID, obs)

    def _invoke_epmem_query(self, obs):
        clip_threshold = MINECLIP_THRESHOLDS[self.cur_task]
        result, stats = self.epmem.query(self.cur_task, clip_threshold)

        if stats is not None:
            self._print(r"[bold]Query stats:[/bold] {}".format(stats))

        if result is None:
            self._mode_switch(AgentMode.EXPLORE, obs)
            return

        pos, cam = result
        if self.nav_target is None or np.linalg.norm(self.nav_target - pos) > 3:
            self.nav_target = pos
            self.cam_target = [x.item() for x in cam]
            self._mode_switch(AgentMode.NAVIGATE, obs)

    def _mode_switch(self, new_mode: AgentMode, obs):
        if new_mode != AgentMode.INVALID:
            new_mode = AgentMode.EXECUTE

        if self.mode != new_mode:
            self.steve1.reset()
            self.vpt_nav.reset()

        self._print(f"Mode switch from {self.mode.name} to {new_mode.name} (runtime={self.mode_count} steps, timestep={self.timestep})")

        location = obs["location_stats"]["pos"]
        cur_loc = np.array([location[0], location[1], location[2]])
        self._print(f"\tCurrent location: {cur_loc}")
        cur_cam = np.array([obs["location_stats"]["pitch"].item(), obs["location_stats"]["yaw"].item()])
        self._print(f"\tCurrent camera: {cur_cam}")

        if new_mode == AgentMode.INVALID:
            self.nav_target = None
            self.cam_target = None
            self.reached = True
            self.reach_time = 0
        elif new_mode == AgentMode.EXECUTE:
            if self.steve1_prompt is None and self.cur_task in SKILLS:
                self.steve1_prompt = SKILLS[self.cur_task]["execute"]
            self._print(f"\tSteve1 prompt: Break the block in front of you and move forward to escape")
            # task
            self.steve1.set_goal("Break the block in front of you and move forward to escape")
            self.nav_target = None
            self.cam_target = None
        elif new_mode == AgentMode.NAVIGATE:
            self._print(f"\tNavigation target: {self.nav_target}")
        elif new_mode == AgentMode.CAM_ADJUST:
            self._print(f"\tCamera target: {self.cam_target}")
        elif new_mode == AgentMode.EXPLORE:
            if self.mode != AgentMode.EXPLORE:
                self.reached = True
                self.reach_time = 0
        else:
            raise ValueError("Invalid mode")

        if new_mode == AgentMode.NAVIGATE:
            self.agent_log.append({
                "mode": new_mode.name,
                "mode_count": self.mode_count,
                "timestep": self.timestep,
                "target": self.nav_target.tolist()
            })
        else:
            self.agent_log.append({
                "mode": new_mode.name,
                "mode_count": self.mode_count,
                "timestep": self.timestep
            })
        if self.eval_tracking_wrapper is not None:
            self.eval_tracking_wrapper.set_agent_mode(new_mode.name)
        self.mode = new_mode
        self.mode_count = 0

    def save_log(self, path):
        with open(path, "wt") as f:
            json.dump(self.agent_log, f, indent=4)
    
    def save_epmem_log(self, path):  
        data = self.epmem.query_logs
        with open(path, "wt") as f:
            json.dump(data, f, indent=4)

    def save_epmem_cluster_log(self, path):
        clusters = self.epmem.memory.clusters
        cluster_poss = {}
        
        if self.epmem.memory_option in ['fifo_memory', 'event_memory', 'place_memory']:
            for i, cluster in enumerate(clusters):
                posyaws = []
                for record in cluster['records']:
                    pos = record.pos
                    yaw = record.cam[1]
                    posyaw = np.concatenate([pos, yaw], axis=0)
                    posyaws.append(posyaw)
                
                if 'center_pos' in cluster.keys():
                    c_pos = cluster['center_pos']
                    c_yaw = cluster['center_yaw']
                    c_posyaw = np.concatenate([c_pos, np.array([c_yaw])], axis=0)
                    posyaws.append(c_posyaw)
                
                cluster_poss[f'{i}'] = np.stack(posyaws, axis=0)
        elif self.epmem.memory_option in ['place_event_memory', 'place_event_memory_center_embed']:
            for j, cluster in enumerate(clusters):
                for i, event_cluster in enumerate(cluster['event_clusters']):
                    posyaws = []
                    for record in event_cluster['records']:
                        pos = record.pos
                        yaw = record.cam[1]
                        posyaw = np.concatenate([pos, yaw], axis=0)
                        posyaws.append(posyaw)
                    
                    c_pos = cluster['center_pos']
                    c_yaw = cluster['center_yaw']
                    c_posyaw = np.concatenate([c_pos, np.array([c_yaw])], axis=0)
                    posyaws.append(c_posyaw)
                    cluster_poss[f'{j}_{i}'] = np.stack(posyaws, axis=0)
        
        np.savez(path, **cluster_poss)

    def save_explore_log(self, path):
        data = self.reach_logs
        with open(path, "wt") as f:
            json.dump(data, f, indent=4)

    def _find_eval_tracking_wrapper(self, env):
        current = env
        depth = 0
        while current is not None and depth < 32:
            if isinstance(current, EvalTrackingWrapper):
                return current
            current = getattr(current, "env", None)
            depth += 1
        return None

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
