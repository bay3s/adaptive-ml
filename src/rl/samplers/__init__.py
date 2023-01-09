import gym
from .multi_task_sampler import MultiTaskSampler
from .sampler import Sampler


def make_env(env_name, env_kwargs={}, seed=None):
    def _make_env():
        env = gym.make(env_name, **env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env


__all__ = ['make_env', 'Sampler', 'MultiTaskSampler']
