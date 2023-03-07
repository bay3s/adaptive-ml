import torch

from rl.utils.functions.optimization_functions import zero_optim_grads
from rl.learners.reinforce import REINFORCE

from rl.structs import EnvSpec
from rl.optimizers import WrappedOptimizer
from rl.networks.value_functions.base_value_function import BaseValueFunction
from rl.networks.policies.base_policy import BasePolicy


class TRPO(REINFORCE):

  def __init__(
    self,
    env_spec: EnvSpec,
    policy: BasePolicy,
    value_function: BaseValueFunction,
    sampler,
    policy_optimizer: WrappedOptimizer,
    vf_optimizer: WrappedOptimizer,
    num_train_per_epoch: int = 1,
    discount: float = 0.99,
    gae_lambda: float = 0.98,
    center_adv: bool = True,
    positive_adv: bool = False,
    policy_ent_coeff: float = 0.0,
    use_softplus_entropy: bool = False,
    stop_entropy_gradient: bool = False,
    entropy_method: str = 'no_entropy'
  ):
    """
    Trust Region Policy Optimization (TRPO).

    Args:
      env_spec (EnvSpec): Environment specification.
      policy (garage.torch.policies.Policy): Policy.
      value_function (garage.torch.value_functions.ValueFunction): The value function.
      sampler (garage.sampler.Sampler): Sampler.
      policy_optimizer (garage.torch.optimizer.WrappedOptimizer): Optimizer for policy.
      vf_optimizer (garage.torch.optimizer.WrappedOptimizer): Optimizer for value function.
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
        surrogate objective. See https://arxiv.org/abs/1805.00909 for details.
    """
    super().__init__(
      env_spec=env_spec,
      policy=policy,
      value_function=value_function,
      sampler=sampler,
      policy_optimizer=policy_optimizer,
      vf_optimizer=vf_optimizer,
      num_train_per_epoch=num_train_per_epoch,
      discount=discount,
      gae_lambda=gae_lambda,
      center_adv=center_adv,
      positive_adv=positive_adv,
      policy_ent_coeff=policy_ent_coeff,
      use_softplus_entropy=use_softplus_entropy,
      stop_entropy_gradient=stop_entropy_gradient,
      entropy_method=entropy_method
    )

  def _compute_objective(self, advantages, obs, actions, rewards):
    """
    Compute objective value.

    Args:
      advantages (torch.Tensor): Advantage value at each step with shape :math:`(N \dot [T], )`.
      obs (torch.Tensor): Observation from the environment with shape :math:`(N \dot [T], O*)`.
      actions (torch.Tensor): Actions fed to the environment with shape :math:`(N \dot [T], A*)`.
      rewards (torch.Tensor): Acquired rewards with shape :math:`(N \dot [T], )`.

    Returns:
      torch.Tensor: Calculated objective values with shape :math:`(N \dot [T], )`.
    """
    with torch.no_grad():
      old_ll = self._old_policy(obs)[0].log_prob(actions)

    new_ll = self.policy(obs)[0].log_prob(actions)
    likelihood_ratio = (new_ll - old_ll).exp()

    surrogate = likelihood_ratio * advantages

    return surrogate

  def _train_policy(self, obs, actions, rewards, advantages):
    """
    Train the policy.

    Args:
      obs (torch.Tensor): Observation from the environment with shape :math:`(N, O*)`.
      actions (torch.Tensor): Actions fed to the environment with shape :math:`(N, A*)`.
      rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
      advantages (torch.Tensor): Advantage value at each step with shape :math:`(N, )`.

    Returns:
      torch.Tensor: Calculated mean scalar value of policy loss (float).
    """
    zero_optim_grads(self._policy_optimizer._optimizer)

    loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)
    loss.backward()

    self._policy_optimizer.step(
      f_loss=lambda: self._compute_loss_with_adv(obs, actions, rewards, advantages),
      f_constraint=lambda: self._compute_kl_constraint(obs)
    )

    return loss
