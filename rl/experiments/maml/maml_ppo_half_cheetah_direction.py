import os
import torch

from rl.envs import HalfCheetahDirectionEnv, NormalizedEnv, GymEnv
from rl.utils.functions.device_functions import set_seed
from rl.samplers.task_samplers.meta_task_sampler import MetaTaskSampler
from rl.samplers.workers import WorkerFactory
from rl.networks.policies import GaussianMLPPolicy
from rl.networks.value_functions import GaussianMLPValueFunction
from rl.utils.training.trainer import Trainer

from rl.metalearners.maml.maml_ppo import MAMLPPO
from rl.utils.training.snapshotter import Snapshotter
from rl.samplers import MultiProcessingSampler


set_seed(1234)
max_episode_length = 100
episodes_per_task = 40
epochs = 300
meta_batch_size = 20

env = NormalizedEnv(
  GymEnv(HalfCheetahDirectionEnv(), max_episode_length=max_episode_length),
  expected_action_scale=10.
)

policy = GaussianMLPPolicy(
  env_spec=env.spec,
  hidden_sizes=[64, 64],
  hidden_nonlinearity=torch.tanh,
  output_nonlinearity=None,
)

value_function = GaussianMLPValueFunction(
  env_spec=env.spec,
  hidden_sizes=[32, 32],
  hidden_nonlinearity=torch.tanh,
  output_nonlinearity=None
)

task_sampler = MetaTaskSampler(
  HalfCheetahDirectionEnv,
  wrapper=lambda env, _: NormalizedEnv(
    GymEnv(env, max_episode_length=max_episode_length),
    expected_action_scale=10.
  )
)

worker_factory = WorkerFactory(max_episode_length = env.spec.max_episode_length)
sampler = MultiProcessingSampler(agents=policy, envs=env, worker_factory = worker_factory)

maml_ppo = MAMLPPO(
  env=env,
  policy=policy,
  sampler=sampler,
  task_sampler=task_sampler,
  value_function=value_function,
  meta_batch_size=meta_batch_size,
  discount=0.99,
  gae_lambda=1.,
  inner_lr=0.1,
  num_grad_updates=1
)

training_snapshotter = Snapshotter(
  snapshot_dir = os.path.join(os.getcwd(), 'data/local/maml-ppo-half-cheetah'),
  snapshot_mode = 'last',
  snapshot_gap = 1
)

trainer = Trainer(snapshotter = training_snapshotter)
trainer.setup(maml_ppo, env)
trainer.train(n_epochs=epochs, batch_size=episodes_per_task * env.spec.max_episode_length)

"""
Next step is to do the evaluation, can use the same configs as either the MAML paper or the PROMP paper.
"""
pass
