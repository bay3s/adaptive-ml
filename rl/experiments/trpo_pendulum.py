import gym
import os
import torch

from rl.utils.functions.device_functions import set_seed
from rl.samplers import LocalSampler
from rl.samplers.workers import WorkerFactory

from rl.learners import TRPO
from rl.networks import (
  GaussianMLPPolicy,
  GaussianMLPValueFunction
)
from rl.optimizers import WrappedOptimizer, ConjugateGradientOptimizer
from rl.envs import GymEnv
from rl.utils.training.trainer import Trainer
from rl.utils.training.snapshotter import Snapshotter


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


worker_factory = WorkerFactory(max_episode_length = env.spec.max_episode_length)
local_sampler = LocalSampler(agents=policy, envs=env, worker_factory = worker_factory)

policy_optimizer = WrappedOptimizer(
  ConjugateGradientOptimizer(policy.parameters(), max_constraint_value = 0.01),
  max_optimization_epochs=10,
  minibatch_size = 64
)

vf_optimizer = WrappedOptimizer(
  torch.optim.Adam(value_function.parameters(), lr=2.5e-4),
  max_optimization_epochs=10,
  minibatch_size = 64
)

trpo = TRPO(
  env_spec=env.spec,
  policy=policy,
  value_function=value_function,
  sampler=local_sampler,
  discount=0.99,
  center_adv=False,
  policy_optimizer = policy_optimizer,
  vf_optimizer = vf_optimizer
)

training_snapshotter = Snapshotter(
  snapshot_dir = os.path.join(os.getcwd(), 'data/local/trpo-pendulum'),
  snapshot_mode = 'last',
  snapshot_gap = 1
)

trainer = Trainer(training_snapshotter)
trainer.setup(trpo, env)
trainer.train(n_epochs=20, batch_size=10_000)
pass
