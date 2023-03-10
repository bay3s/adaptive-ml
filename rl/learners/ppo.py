import torch

from rl.learners.reinforce import REINFORCE

from rl.structs import EnvSpec
from rl.optimizers import WrappedOptimizer
from rl.networks.value_functions.base_value_function import BaseValueFunction
from rl.networks.policies.base_policy import BasePolicy


class PPO(REINFORCE):

  def __init__(
    self,
    env_spec: EnvSpec,
    policy: BasePolicy,
    value_function: BaseValueFunction,
    sampler,
    policy_optimizer: WrappedOptimizer,
    vf_optimizer: WrappedOptimizer,
    lr_clip_range = 2e-1,
    num_train_per_epoch = 1,
    discount = 0.99,
    gae_lambda = 0.97,
    center_adv = True,
    positive_adv = False,
    policy_ent_coeff = 0.0,
    use_softplus_entropy = False,
    stop_entropy_gradient = False,
    entropy_method = 'no_entropy'
  ):
    """
    Proximal Policy Optimization (PPO).

    Args:
      env_spec (EnvSpec): Environment specification.
      policy (BasePolicy): Policy.
      value_function (ValueFunction): The value function.
      sampler (Sampler): Sampler.
      policy_optimizer (WrappedOptimizer): Optimizer for policy.
      vf_optimizer (WrappedOptimizer): Optimizer for value function.
      lr_clip_range (float): The limit on the likelihood ratio between policies.
      num_train_per_epoch (int): Number of train_once calls per epoch.
      discount (float): Discount.
      gae_lambda (float): Lambda used for generalized advantage estimation.
      center_adv (bool): Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
      positive_adv (bool): Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
      policy_ent_coeff (float): The coefficient of the policy entropy. Setting it to zero would mean no entropy
        regularization.
      use_softplus_entropy (bool): Whether to estimate the softmax distribution of the entropy to prevent the entropy
        from being negative.
      stop_entropy_gradient (bool): Whether to stop the entropy gradient.
      entropy_method (str): A string from: 'max', 'regularized', 'no_entropy'. The type of entropy method to use.
        'max' adds the dense entropy to the reward for each time step. 'regularized' adds the mean entropy to the
        surrogate objective. See https://arxiv.org/abs/1805.00909 for more details.
    """
    super().__init__(
      env_spec = env_spec,
      policy = policy,
      value_function = value_function,
      sampler = sampler,
      policy_optimizer = policy_optimizer,
      vf_optimizer = vf_optimizer,
      num_train_per_epoch = num_train_per_epoch,
      discount = discount,
      gae_lambda = gae_lambda,
      center_adv = center_adv,
      positive_adv = positive_adv,
      policy_ent_coeff = policy_ent_coeff,
      use_softplus_entropy = use_softplus_entropy,
      stop_entropy_gradient = stop_entropy_gradient,
      entropy_method = entropy_method
    )

    self._lr_clip_range = lr_clip_range
    pass

  def _compute_objective(self, advantages, obs, actions, rewards):
    """
    Compute objective value.

    Args:
      advantages (torch.Tensor): Advantage value at each step.
      obs (torch.Tensor): Observation from the environment.
      actions (torch.Tensor): Actions fed to the environment.
      rewards (torch.Tensor): Acquired rewards with shape.

    Returns:
      torch.Tensor: Calculated objective values with shape.
    """
    with torch.no_grad():
      old_ll = self._old_policy(obs)[0].log_prob(actions)

    new_ll = self.policy(obs)[0].log_prob(actions)
    likelihood_ratio = (new_ll - old_ll).exp()

    # compute the surrogate loss.
    surrogate = likelihood_ratio * advantages
    likelihood_ratio_clip = torch.clamp(likelihood_ratio, min = 1 - self._lr_clip_range, max = 1 + self._lr_clip_range)

    # compute the surrogate clip.
    surrogate_clip = likelihood_ratio_clip * advantages

    return torch.min(surrogate, surrogate_clip)