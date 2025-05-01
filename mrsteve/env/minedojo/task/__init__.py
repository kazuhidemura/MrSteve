from omegaconf import OmegaConf

from .minedojo import make_minedojo


CUSTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("mrsteve/env/minedojo/task/minedojo/task_specs.yaml"))


def get_specs(task, **kwargs):
    assert task in CUSTOM_TASK_SPECS

    yaml_specs = CUSTOM_TASK_SPECS[task].copy()
    task_id = yaml_specs.pop("task_id", task)

    sim_specs = yaml_specs.pop("sim_specs", dict())

    task_specs = dict()
    task_specs.update(**yaml_specs)
    task_specs.update(**kwargs)

    return task_id, task_specs, sim_specs


def make(episode_id, task: str, **kwargs):
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)

    env = make_minedojo(episode_id, task_id, task_specs, sim_specs)

    return env
