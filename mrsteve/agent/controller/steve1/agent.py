import pickle
import torch
import json

from mrsteve.lib.steve1.config import PRIOR_INFO
from mrsteve.lib.steve1.data.text_alignment.vae import load_vae_model
from mrsteve.lib.steve1.utils.embed_utils import get_prior_embed
from mrsteve.lib.steve1.MineRLConditionalAgent import MineRLConditionalAgent


with open("mrsteve/data/steve1_skills.json", "rt") as f:
    SKILLS = json.load(f)

SKILL_TARGET_ITEMS = list(SKILLS.keys())
SKILL_EXPLORE_PROMPTS = [SKILLS[skill]["explore"] for skill in SKILL_TARGET_ITEMS]
SKILL_EXECUTE_PROMPTS = [SKILLS[skill]["execute"] for skill in SKILL_TARGET_ITEMS]


class Steve1Agent:
    def __init__(
        self,
        model: str,
        model_weights: str,
        prior_weights: str,
        cond_scale: float,
        mineclip,
        device: torch.device
    ):
        self.mineclip = mineclip
        PRIOR_INFO["model_path"] = prior_weights
        self.prior = load_vae_model(PRIOR_INFO, device)
        self.device = device

        with open(model, "rb") as f:
            agent_parameters = pickle.load(f)

        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

        self.agent = MineRLConditionalAgent(device, policy_kwargs, pi_head_kwargs)
        self.agent.policy.eval()
        self.agent.load_weights(model_weights)

        self.cond_scale = cond_scale
        self.goal_embed = None

    def reset(self):
        self.agent.reset(self.cond_scale)

    def set_goal(self, goal: torch.Tensor | str):
        if isinstance(goal, str):
            self.goal_embed = get_prior_embed(goal, self.mineclip, self.prior, self.agent.device)
        else:
            self.goal_embed = goal

    def get_action(self, obs: torch.Tensor, greedy: bool = False):
        with torch.cuda.amp.autocast():
            minerl_action = self.agent.get_action({"pov": obs}, self.goal_embed, greedy=greedy)
        return minerl_action
