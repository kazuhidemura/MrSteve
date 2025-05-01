import torch
import os

from mrsteve.env.common.types import EvalInfo
from mrsteve.env.minedojo.task.minedojo.wrappers import MinedojoVPTWrapper
from mrsteve.agent.controller.mrsteve.agent import MrSteveAgent
from mrsteve.agent.controller.mrsteve.lib.epmem import EpisodicMemoryWrapper
from mrsteve.lib.steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig


@torch.no_grad()
def main(config, env_entry, eval_info: EvalInfo, logdir: str):
    device = torch.device("cuda:0")
    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()

    env = env_entry(eval_info=eval_info)
    env = EpisodicMemoryWrapper(config.agent.epmem, env, mineclip, device)

    agent = MrSteveAgent(env, config.agent, mineclip, device)

    env = MinedojoVPTWrapper(env)

    done = False
    obs = env.reset()

    while not done:
        action = agent.get_action(obs)
        obs, _, done, _ = env.step(action, use_minerl_action=True)

    # save logs
    agent_log_root = os.path.join(logdir, "agent_log")
    os.makedirs(agent_log_root, exist_ok=True)
    agent.save_log(os.path.join(agent_log_root, f"agent_log_{eval_info.episode_id:03d}.json"))

    epmem_log_root = os.path.join(logdir, "episodic_memory_log")
    os.makedirs(epmem_log_root, exist_ok=True)
    agent.save_epmem_log(os.path.join(epmem_log_root, f"epmem_log_{eval_info.episode_id:03d}.json"))

    epmem_cluster_log_root = os.path.join(logdir, "episodic_memory_cluster_log")
    os.makedirs(epmem_cluster_log_root, exist_ok=True)
    agent.save_epmem_cluster_log(os.path.join(epmem_cluster_log_root, f"epmem_cluster_log_{eval_info.episode_id:03d}.npz"))

    explore_log_root = os.path.join(logdir, "explore_log")
    os.makedirs(explore_log_root, exist_ok=True)
    agent.save_explore_log(os.path.join(explore_log_root, f"explore_log_{eval_info.episode_id:03d}.json"))

    env.close()
