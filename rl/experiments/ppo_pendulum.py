import gym
import torch

from rl.utils.functions.device_functions import set_seed
from rl.samplers import LocalSampler

from rl.learners import PPO
from rl.networks import (
  GaussianMLPPolicy,
  GaussianMLPValueFunction
)
from rl.optimizers import WrappedOptimizer

from rl.envs import GymEnv


set_seed(seed = 1)
env = GymEnv(gym.make('InvertedDoublePendulum-v2'))

policy = GaussianMLPPolicy(
  env.spec,
  hidden_sizes=[64, 64],
  hidden_nonlinearity=torch.tanh,
  output_nonlinearity=None
)

value_function = GaussianMLPValueFunction(
  env_spec=env.spec,
  hidden_sizes=[32, 32],
  hidden_nonlinearity=torch.tanh,
  output_nonlinearity=None
)

# @todo sampling / multiprocessing / etc.
sampler = LocalSampler(
  agents=policy,
  envs=env,
  max_episode_length=env.spec.max_episode_length
)

policy_optimizer = WrappedOptimizer(
  torch.optim.Adam(policy.parameters(), lr=2.5e-4),
  max_optimization_epochs=10,
  minibatch_size = 64
)

vf_optimizer = WrappedOptimizer(
  torch.optim.Adam(value_function.parameters(), lr=2.5e-4),
  max_optimization_epochs=10,
  minibatch_size = 64
)

algo = PPO(
  env_spec=env.spec,
  policy=policy,
  value_function=value_function,
  sampler=None,
  discount=0.99,
  center_adv=False,
  policy_optimizer = policy_optimizer,
  vf_optimizer = vf_optimizer
)

# trainer.setup(algo, env)
# trainer.train(n_epochs=100, batch_size=10000)
