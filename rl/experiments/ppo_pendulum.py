import torch
from rl.utils.functions.device_functions import set_seed

from rl.learners import PPO
from rl.networks import (
  GaussianMLPPolicy,
  GaussianMLPValueFunction
)

from rl.envs import GymEnv


set_seed(seed = 1)
env = GymEnv('InvertedDoublePendulum-v2')

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

# sampler = RaySampler(
#   agents=policy,
#   envs=env,
#   max_episode_length=env.spec.max_episode_length
# )

# @todo policy optimizer, value optimizer, and sampler need to be set.
algo = PPO(
  env_spec=env.spec,
  policy=policy,
  value_function=value_function,
  sampler=None,
  discount=0.99,
  center_adv=False,
  policy_optimizer = None,
  vf_optimizer = None
)

# trainer.setup(algo, env)
# trainer.train(n_epochs=100, batch_size=10000)
