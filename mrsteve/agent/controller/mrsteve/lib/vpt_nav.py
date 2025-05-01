import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from gym3.types import DictType
from mineclip.utils import build_mlp
from typing import Optional

from mrsteve.lib.VPT.agent import (
    MineRLAgent, resize_image, AGENT_RESOLUTION,
    default_device_type, set_default_torch_device, CameraHierarchicalMapping,
    ActionTransformer, ACTION_TRANSFORMER_KWARGS, POLICY_KWARGS, PI_HEAD_KWARGS)
from mrsteve.lib.VPT.lib.scaled_mse_head import ScaledMSEHead
from mrsteve.lib.VPT.lib.tree_util import tree_map
from mrsteve.lib.steve1.embed_conditioned_policy import make_action_head
from mrsteve.lib.VPT.lib.policy import MinecraftPolicy


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class GoalEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 512,
        hidden_depth: int = 3
    ):
        super().__init__()

        self._mlp = build_mlp(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )
    
    def forward(self, goal, compass):
        goal = symlog(goal)
        x = torch.cat([goal, compass], dim=-1)
        return self._mlp(x)


class PointMinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs):
        super().__init__()

        self.net = MinecraftPolicy(**policy_kwargs)
        self.action_space = action_space

        self.goal_embed = GoalEmbedding()
        self.goal_proj = nn.Linear(512, self.net.output_latent_size())
        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)

        # disable gradients what we don't want to update
        for param in self.net.parameters():
            param.requires_grad = False

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def forward(self, obs, first: torch.Tensor, state_in, uncond: torch.Tensor):
        assert isinstance(obs, dict)
        obs = obs.copy()
        mask = obs.pop("mask", None)

        goal_embed = self.goal_embed(obs["goal"], obs["compass"])

        (pi_h, v_h), state_out = self.net(obs, state_in, context={"first": first})

        goal_embed = self.goal_proj(goal_embed)

        pi_logits = self.pi_head(pi_h + goal_embed, mask=mask)
        vpred = self.value_head(v_h + goal_embed)

        return (pi_logits, vpred, None), state_out

    def get_logprob_of_action(self, pd, action):
        """
        Get logprob of taking action `action` given probability distribution
        (see `get_gradient_for_action` to get this distribution)
        """
        log_prob = self.pi_head.logprob(action, pd)

        assert not torch.isnan(log_prob).any()
        return log_prob

    def get_kl_of_action_dists(self, pd1, pd2):
        """
        Get the KL divergence between two action probability distributions
        """
        return self.pi_head.kl_divergence(pd1, pd2)

    def get_output_for_observation(self, obs, state_in, first, uncond):
        """
        Return gradient-enabled outputs for given observation.

        Use `get_logprob_of_action` to get log probability of action
        with the given probability distribution.

        Returns:
          - probability distribution given observation
          - value prediction for given observation
          - new state
        """
        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in, uncond=uncond)
                
        return pd, self.value_head.denormalize(vpred)[:, 0], state_out

    @torch.no_grad()
    def act(self, obs, first, state_in, stochastic: bool = True,
            taken_action=None, return_pd=False, cond_scale=None):
        if cond_scale is not None:
            B = obs["img"].shape[0]
            assert B == 1, "cond_scale only works for batch size 1"
            # Change the batch size fo 2, and duplicate the first element.
            obs = tree_map(lambda x: torch.cat([x, x], dim=0), obs)
            first = torch.cat([first, first], dim=0)
            uncond = torch.tensor([0, 1], device=obs["img"].device).view(2, 1)
        else:
            uncond = torch.tensor([0], device=obs["img"].device).view(1, 1)
        
        obs = tree_map(lambda x: x.unsqueeze(1), obs)
        first = first.unsqueeze(1)

        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in, uncond=uncond)

        # Compute entropy of the action distribution (buttons only)
        buttons = pd["buttons"][0, 0, 0, :]
        softmax_buttons = torch.softmax(buttons, dim=-1)
        self.entropy_last = -torch.sum(softmax_buttons * torch.log(softmax_buttons), dim=-1)

        if cond_scale is not None:
            # Combine the pytree elements using a weighted sum across the batch.
            # x[0]: conditional
            # x[1]: unconditional
            # cond_scale = 0: regular conditional policy
            # cond_scale > 0: subtract some of the unconditional policy
            pd = tree_map(lambda x: (((1 + cond_scale) * x[0]) - (cond_scale * x[1])).unsqueeze(0), pd)

        if taken_action is None:
            ac = self.pi_head.sample(pd, deterministic=not stochastic)
        else:
            ac = tree_map(lambda x: x.unsqueeze(1), taken_action)
        log_prob = self.pi_head.logprob(ac, pd)
        assert not torch.isnan(log_prob).any()
    
        # After unsqueezing, squeeze back to remove fictitious time dimension
        result = {"log_prob": log_prob[:, 0], "vpred": self.value_head.denormalize(vpred)[:, 0]}
        if return_pd:
            result["pd"] = tree_map(lambda x: x[:, 0], pd)
        ac = tree_map(lambda x: x[:, 0], ac)

        return ac, state_out, result


class PointConditionalAgent(MineRLAgent):
    def __init__(self, device=None, policy_kwargs=None, pi_head_kwargs=None):
        if device is None:
            device = default_device_type()
        self.device = torch.device(device)
        set_default_torch_device(self.device)

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            action_space=action_space,
        )

        self.mean_delta_len = None
        self.prev_pos = None

        self.policy = PointMinecraftAgentPolicy(**agent_kwargs).to(device)
        self.reset(cond_scale=None)
        self._dummy_first = torch.from_numpy(np.array((False,))).to(device)

    def reset(self, cond_scale=None):
        if cond_scale is None:
            self.hidden_state = self.policy.initial_state(1)
        else:
            self.hidden_state = self.policy.initial_state(2)
        self.cond_scale = cond_scale

        self.mean_delta_len = None
        self.prev_pos = None

    def get_agent_input_pov(self, frame: np.ndarray):
        agent_input_pov = resize_image(frame, AGENT_RESOLUTION)[None]
        return agent_input_pov
    
    def get_action(self, minerl_obs, goal_pos, greedy=False):
        agent_input = self._env_obs_to_agent(minerl_obs, goal_pos)

        force_jump = False
        cur_loc = minerl_obs["location_stats"]["pos"]
        cur_loc = np.array([cur_loc[0], cur_loc[2]])
        if self.prev_pos is not None:
            delta_len = np.linalg.norm(cur_loc - self.prev_pos)

            if self.mean_delta_len is None:
                self.mean_delta_len = delta_len
            else:
                if self.mean_delta_len < 1e-3:
                    force_jump = True
                
                alpha = 0.001 ** (1/20)
                self.mean_delta_len = alpha * self.mean_delta_len + (1 - alpha) * delta_len
        self.prev_pos = cur_loc

        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=not greedy, cond_scale=self.cond_scale
        )
        minerl_action = self._agent_action_to_env(agent_action)

        if force_jump:
            minerl_action["jump"] = np.array([1])
        return minerl_action
    
    def _env_obs_to_agent(self, minerl_obs, goal_pos, device=None):
        if device is None:
            device = self.device

        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": torch.from_numpy(agent_input).to(device)}

        cur_loc = minerl_obs["location_stats"]["pos"]
        cur_loc = np.array([cur_loc[0], cur_loc[2]])
        goal = goal_pos - cur_loc
                
        goal_length = np.linalg.norm(goal)
        goal_norm_factor = 20 / np.clip(goal_length, a_min=20, a_max=None)
        goal = goal * goal_norm_factor

        cur_pitch, cur_yaw = minerl_obs["location_stats"]["pitch"], minerl_obs["location_stats"]["yaw"]
        cur_pitch, cur_yaw = np.deg2rad(cur_pitch), np.deg2rad(cur_yaw)
        compass = np.concatenate([
            np.cos(cur_yaw), np.sin(cur_yaw), np.cos(cur_pitch), np.sin(cur_pitch)
        ], axis=-1)

        agent_input["goal"] = torch.from_numpy(goal).float().unsqueeze(0).to(device)
        agent_input["compass"] = torch.from_numpy(compass).float().unsqueeze(0).to(device)
        return agent_input

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False, device=None):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        if device is None:
            device = self.device

        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: torch.from_numpy(v).to(device) for k, v in action.items()}
        return action
