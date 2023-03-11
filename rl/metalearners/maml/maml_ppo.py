import torch

from rl.metalearners.maml.base_maml import BaseMAML
from rl.optimizers import WrappedOptimizer
from rl.learners import PPO


class MAMLPPO(BaseMAML):

  def __init__(
    self,
    env,
    policy,
    value_function,
    sampler,
    task_sampler,
    inner_lr = 1e-1,
    outer_lr = 1e-3,
    lr_clip_range = 5e-1,
    discount = 0.99,
    gae_lambda = 1.0,
    center_adv = True,
    positive_adv = False,
    policy_ent_coeff = 0.0,
    use_softplus_entropy = False,
    stop_entropy_gradient = False,
    entropy_method = 'no_entropy',
    meta_batch_size = 20,
    num_grad_updates = 1,
    meta_evaluator = None,
    evaluate_every_n_epochs = 1
  ):
    policy_optimizer = WrappedOptimizer(torch.optim.Adam(params = policy.parameters(), lr = inner_lr))
    vf_optimizer = WrappedOptimizer(torch.optim.Adam(params = value_function.parameters(), lr = inner_lr))

    inner_algo = PPO(
      env.spec,
      policy,
      value_function,
      None,
      policy_optimizer = policy_optimizer,
      vf_optimizer = vf_optimizer,
      lr_clip_range = lr_clip_range,
      num_train_per_epoch = 1,
      discount = discount,
      gae_lambda = gae_lambda,
      center_adv = center_adv,
      positive_adv = positive_adv,
      policy_ent_coeff = policy_ent_coeff,
      use_softplus_entropy = use_softplus_entropy,
      stop_entropy_gradient = stop_entropy_gradient,
      entropy_method = entropy_method
    )

    meta_optimizer = torch.optim.Adam(params = policy.parameters(), lr = outer_lr, eps = 1e-5)
    super().__init__(
      inner_algo = inner_algo,
      env = env,
      policy = policy,
      sampler = sampler,
      task_sampler = task_sampler,
      meta_optimizer = meta_optimizer,
      meta_batch_size = meta_batch_size,
      inner_lr = inner_lr,
      num_grad_updates = num_grad_updates,
      meta_evaluator = meta_evaluator,
      evaluate_every_n_epochs = evaluate_every_n_epochs
    )
    pass
