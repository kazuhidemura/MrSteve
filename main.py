import argparse
import multiprocessing as mp
import numpy as np
import hydra
import logging
from functools import partial
from hydra.core.hydra_config import HydraConfig

from mrsteve.env.common.types import EvalInfo
from mrsteve.env.minedojo import env_main as minedojo_env_main
from mrsteve.utils import get_function


def _eval_worker(id_cluster: np.array, config, env_entry, agent_entry, logdir, is_debug):
        for episode_id in id_cluster:
            eval_info = EvalInfo(episode_id=episode_id)

            while True:
                try:
                    agent_entry(config, env_entry, eval_info, logdir)
                    break
                except Exception as e:
                    if is_debug:
                        raise e
                    else:
                        print(f"Error in episode {episode_id}: {e}")


def eval_helper(config, agent_entry, env_entry, logdir: str):
    if config.n_workers == 1:
        cluster = np.arange(config.n_episodes)
        _eval_worker(cluster, config, env_entry, agent_entry, logdir, config.debug)
    else:
        mp.set_start_method("spawn")

        episode_ids = np.arange(config.n_episodes)
        id_clusters = np.array_split(episode_ids, config.n_workers)

        workers = []
        for cluster in id_clusters:
            worker = mp.Process(target=_eval_worker, args=(cluster, config, env_entry, agent_entry, logdir, config.debug))
            worker.start()
            workers.append(worker)
        
        for worker in workers:
            worker.join()


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(config):
    logdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output directory: {logdir}")

    agent_entry = get_function(config.agent.eval_entry)
    env_entry = partial(minedojo_env_main, config=config, logdir=logdir)

    eval_helper(config, agent_entry, env_entry, logdir)


if __name__ == "__main__":
    main()