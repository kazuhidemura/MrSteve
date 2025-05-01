import argparse
import glob
import os
import numpy as np
from scipy.stats import sem
from tabulate import tabulate


def get_stats(logdir):
    filelist = glob.glob(os.path.join(logdir, "episode", "*.npz"))
    n_episodes = len(filelist)

    successes = []
    ep_lens = []
    for filename in filelist:
        data = np.load(filename)

        total_tasks = data["total_tasks"]
        success_tasks = data["success_tasks"]
        successes.append(total_tasks == success_tasks)

        episode_len = data["reward"].shape[0]
        ep_lens.append(episode_len)

    success_rate = np.mean(successes)
    ep_len_mean = np.mean(ep_lens)
    ep_len_sem = sem(ep_lens)

    return n_episodes, success_rate, (ep_len_mean, ep_len_sem)


def main(args):
    table = []
    for logdir in args.rootdir:
        n_episodes, success_rate, (ep_len_mean, ep_len_sem) = get_stats(logdir)
        ep_len_str = f"{ep_len_mean:.2f} \pm {ep_len_sem:.2f}"
        table.append([logdir, n_episodes, success_rate, ep_len_str])

    print(tabulate(table, headers=["logdir", "n_episodes", "success rate", "episode length"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", type=str, nargs="+")
    args = parser.parse_args()

    main(args)
