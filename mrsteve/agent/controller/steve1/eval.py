import torch

from mrsteve.env.common.types import EvalInfo
from mrsteve.env.minedojo.task.minedojo.wrappers import MinedojoVPTWrapper
from mrsteve.agent.controller.steve1.agent import Steve1Agent, SKILLS
from mrsteve.lib.steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig


@torch.no_grad()
def main(config, env_entry, eval_info: EvalInfo, logdir: str):
    env = env_entry(eval_info=eval_info)
    env = MinedojoVPTWrapper(env)

    device = torch.device("cuda:0")

    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()

    agent = Steve1Agent(
        config.agent.model,
        config.agent.weights,
        config.agent.prior_weights,
        config.agent.cond_scale,
        mineclip,
        device,
    )
    agent.reset()

    prev_task = None
    greedy_steve1 = config.agent.greedy
    
    obs = env.reset()
    done = False

    while not done:
        if prev_task != obs["task"]:
            prev_task = obs["task"]
            agent.set_goal(SKILLS[prev_task]["execute"])

        minerl_action = agent.get_action(obs["rgb"], greedy=greedy_steve1)
        obs, _, done, _ = env.step(minerl_action, use_minerl_action=True)

    env.close()
