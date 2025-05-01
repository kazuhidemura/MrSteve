from typing import Optional

from ..common.wrappers import ActionTrackingWrapper
from .wrappers import EvalTrackingWrapper
from .task import make
from mrsteve.env.common.types import EvalInfo


def env_main(config,
             logdir: str,
             eval_info: Optional[EvalInfo] = None):
    env = make(eval_info.episode_id, config.task)

    env = EvalTrackingWrapper(env, logdir, eval_info)
    env = ActionTrackingWrapper(env, logdir, eval_info)

    return env
