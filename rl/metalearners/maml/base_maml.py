import collections
from typing import Tuple, List
import copy
import numpy as np
import torch
from dowel import tabular

from rl.envs import GymEnv
from rl.networks.policies.base_policy import BasePolicy
from rl.samplers.trajectory_samplers.base_sampler import BaseSampler
from rl.learners import REINFORCE
from rl.utils.training.trainer import Trainer

from rl.utils.functions.rl_functions import discount_cumsum
from rl.utils.functions.preprocessing_functions import np_to_torch
from rl.utils.functions.optimization_functions import zero_optim_grads, update_module_params
from rl.utils.functions.logging_functions import log_multitask_performance
from rl.structs import EpisodeBatch

from rl.optimizers import (
  DifferentiableSGD,
  ConjugateGradientOptimizer
)


class BaseMAML:

  def __init__(self, env: GymEnv, policy: BasePolicy, sampler: BaseSampler, inner_algo: REINFORCE, task_sampler,
               meta_optimizer, meta_batch_size: int, inner_lr: float, num_grad_updates: int,
               meta_evaluator, evaluate_every_n_epochs: int):
    """
    Model-Agnostic Meta-Learning.

    Args:
      env (GymEnv): Gym environment to meta-learn on.
      policy (BasePolicy): Policy.
      sampler (BaseSampler): Sampler to use.
      inner_algo (REINFORCE): Policy gradient algorithm.
      task_sampler (TaskSampler): Task sampler.
      meta_optimizer (WrappedOptmizer): Optimizer for meta-learning.
      meta_batch_size (int): Number of tasks sampled per batch.
      inner_lr (float): Adaptation learning rate.
      num_grad_updates (int): Number of adaptation gradient steps.
      meta_evaluator (MetaEvaluator): A meta evaluator for meta-testing.
      evaluate_every_n_epochs (int): Do meta-testing every epoch.
    """
    self._sampler = sampler

    self.max_episode_length = inner_algo.max_episode_length
    self._meta_evaluator = meta_evaluator
    self._policy = policy
    self._env = env
    self._task_sampler = task_sampler
    self._value_function = copy.deepcopy(inner_algo._value_function)
    self._initial_vf_state = self._value_function.state_dict()
    self._num_grad_updates = num_grad_updates
    self._meta_batch_size = meta_batch_size
    self._inner_algo = inner_algo
    self._inner_optimizer = DifferentiableSGD(self._policy, lr = inner_lr)
    self._inner_lr = inner_lr
    self._meta_optimizer = meta_optimizer
    self._evaluate_every_n_epochs = evaluate_every_n_epochs
    pass

  def train(self, trainer: Trainer) -> float:
    """
    Obtain samples and start training for each epoch.

    Args:
      trainer (Trainer): Gives the lgorithm access to the Trainer's internal functions.

    Returns:
      float
    """
    last_return = None

    for _ in trainer.epochs():
      all_samples, all_params = self._obtain_samples(trainer)
      last_return = self._train_once(trainer, all_samples, all_params)
      trainer.step_itr += 1

    return last_return

  def _train_once(self, trainer: Trainer, all_samples: List, all_params: List) -> float:
    """
    Trains the algorithm once and gives the mean return.

    Args:
      trainer (Trainer): The experiment runner.
      all_samples (List[List[MAMLEpisodeBatch]]): 2-d list of MAMLEpisodeBatch.
      all_params (List[dict]): A list of named parameter dictionaries.

    Returns:
      float
    """
    itr = trainer.step_itr
    old_theta = dict(self._policy.named_parameters())

    kl_before = self._compute_kl_constraint(all_samples, all_params, set_grad = False)
    meta_objective = self._compute_meta_loss(all_samples, all_params)

    zero_optim_grads(self._meta_optimizer)
    meta_objective.backward()

    self._meta_optimize(all_samples, all_params)

    loss_after = self._compute_meta_loss(all_samples, all_params, set_grad = False)
    kl_after = self._compute_kl_constraint(all_samples, all_params, set_grad = False)

    with torch.no_grad():
      policy_entropy = self._compute_policy_entropy([task_samples[0] for task_samples in all_samples])
      average_return = self._log_performance(
        itr,
        all_samples,
        meta_objective.item(),
        loss_after.item(),
        kl_before.item(),
        kl_after.item(),
        policy_entropy.mean().item()
      )

    # @todo add a proper evaluation step.
    if self._meta_evaluator and itr % self._evaluate_every_n_epochs == 0:
      self._meta_evaluator.evaluate(self)

    update_module_params(self._old_policy, old_theta)

    return average_return

  def _train_value_function(self, paths: List) -> torch.Tensor:
    """
    Train the value function.

    Args:
      paths (list[dict]): A list of collected paths.

    Returns:
      torch.Tensor
    """
    # MAML resets a value function to its initial state before training.
    self._value_function.load_state_dict(self._initial_vf_state)

    obs = np.concatenate([path['observations'] for path in paths], axis = 0)
    returns = np.concatenate([path['returns'] for path in paths])

    obs = np_to_torch(obs)
    returns = np_to_torch(returns.astype(np.float32))

    vf_loss = self._value_function.compute_loss(obs, returns)
    zero_optim_grads(self._inner_algo._vf_optimizer._optimizer)
    vf_loss.backward()
    self._inner_algo._vf_optimizer.step()

    return vf_loss

  def _obtain_samples(self, trainer: Trainer) -> Tuple:
    """
    Obtain samples for each task before and after the fast adaptation.

    Args:
      trainer (Trainer): A trainer instance to obtain samples.

    Returns:
      Tuple
    """
    tasks = self._task_sampler.sample(self._meta_batch_size)
    theta = dict(self._policy.named_parameters())

    all_samples = [[] for _ in range(len(tasks))]
    all_params = []

    for i, env_up in enumerate(tasks):

      for j in range(self._num_grad_updates + 1):
        episodes = trainer.obtain_episodes(self.exploration_policy, env_update = env_up)
        batch_samples = self._process_samples(episodes)
        all_samples[i].append(batch_samples)

        if j < self._num_grad_updates:
          require_grad = j < self._num_grad_updates - 1
          self._adapt(batch_samples, set_grad = require_grad)

      all_params.append(dict(self._policy.named_parameters()))
      update_module_params(self._policy, theta)

    return all_samples, all_params

  def _adapt(self, batch_samples, set_grad: bool = True):
    """
    Performs one MAML inner step to update the policy.

    Args:
      batch_samples (_MAMLEpisodeBatch): Samples data for one task and one gradient step.
      set_grad (bool): if False, update policy parameters in-place.
    """
    loss = self._inner_algo._compute_loss(*batch_samples[1:])

    # update policy parameters with one SGD step
    self._inner_optimizer.set_grads_none()
    loss.backward(create_graph = set_grad)

    with torch.set_grad_enabled(set_grad):
      self._inner_optimizer.step()

  def _meta_optimize(self, all_samples, all_params):
    """
    Meta-optimization step.

    Args:
      all_samples (List[List[MAMLEpisodeBatch]]): 2-d list of MAMLEpisodeBatch.
      all_params (List[dict]): A list of named parameter dictionaries.

    Returns:
      float
    """
    if isinstance(self._meta_optimizer, ConjugateGradientOptimizer):
      self._meta_optimizer.step(
        f_loss = lambda: self._compute_meta_loss(all_samples, all_params, set_grad = False),
        f_constraint = lambda: self._compute_kl_constraint(all_samples, all_params)
      )
    else:
      self._meta_optimizer.step(lambda: self._compute_meta_loss(all_samples, all_params, set_grad = False))

  def _compute_meta_loss(self, all_samples, all_params, set_grad = True):
    """
    Compute loss to meta-optimize.

    Args:
      all_samples (list[list[_MAMLEpisodeBatch]]): A two dimensional list of _MAMLEpisodeBatch/
      all_params (list[dict]): A list of named parameter dictionaries.
      set_grad (bool): Whether to enable gradient calculation or not.

    Returns:
      torch.Tensor: Calculated mean value of loss.
    """
    theta = dict(self._policy.named_parameters())
    old_theta = dict(self._old_policy.named_parameters())

    losses = []
    for task_samples, task_params in zip(all_samples, all_params):
      for i in range(self._num_grad_updates):
        require_grad = i < self._num_grad_updates - 1 or set_grad
        self._adapt(task_samples[i], set_grad = require_grad)

      update_module_params(self._old_policy, task_params)
      with torch.set_grad_enabled(set_grad):
        last_update = task_samples[-1]
        loss = self._inner_algo._compute_loss(*last_update[1:])

      losses.append(loss)

      update_module_params(self._policy, theta)
      update_module_params(self._old_policy, old_theta)

    return torch.stack(losses).mean()

  def _compute_kl_constraint(self, all_samples, all_params, set_grad = True):
    """
    Compute KL divergence.

    Args:
      all_samples (list[list[_MAMLEpisodeBatch]]): Two dimensional list of _MAMLEpisodeBatch.
      all_params (list[dict]): A list of named parameter dictionaries.
      set_grad (bool): Whether to enable gradient calculation or not.

    Returns:
      torch.Tensor
    """
    theta = dict(self._policy.named_parameters())
    old_theta = dict(self._old_policy.named_parameters())

    kls = []
    for task_samples, task_params in zip(all_samples, all_params):
      for i in range(self._num_grad_updates):
        require_grad = i < self._num_grad_updates - 1 or set_grad
        self._adapt(task_samples[i], set_grad = require_grad)

      update_module_params(self._old_policy, task_params)

      with torch.set_grad_enabled(set_grad):
        kl = self._inner_algo._compute_kl_constraint(task_samples[-1].observations)
        pass

      kls.append(kl)

      update_module_params(self._policy, theta)
      update_module_params(self._old_policy, old_theta)

    return torch.stack(kls).mean()

  def _compute_policy_entropy(self, task_samples):
    """
    Compute policy entropy.

    Args:
      task_samples (list[_MAMLEpisodeBatch]): Samples data for one task.

    Returns:
      torch.Tensor
    """
    obs = torch.cat([samples.observations for samples in task_samples])
    entropies = self._inner_algo._compute_policy_entropy(obs)

    return entropies.mean()

  def _process_samples(self, episodes):
    """
    Process sample data based on the collected paths.

    Args:
      episodes (EpisodeBatch): Collected batch of episodes.

    Returns:
      _MAMLEpisodeBatch
    """
    paths = episodes.to_list()
    for path in paths:
      path['returns'] = discount_cumsum(path['rewards'], self._inner_algo.discount).copy()

    self._train_value_function(paths)

    obs = torch.Tensor(episodes.padded_observations)
    actions = torch.Tensor(episodes.padded_actions)
    rewards = torch.Tensor(episodes.padded_rewards)
    valids = torch.Tensor(episodes.lengths).int()

    with torch.no_grad():
      baselines = self._inner_algo._value_function(obs)

    return _MAMLEpisodeBatch(paths, obs, actions, rewards, valids, baselines)

  def adapt_policy(self, exploration_policy: BasePolicy, exploration_episodes: EpisodeBatch):
    """
    Adapt the policy by one gradient steps for a task.

    Args:
      exploration_policy (Policy): A policy which was returned from get_exploration_policy(), and which generated
        exploration_episodes by interacting with an environment.
      exploration_episodes (EpisodeBatch): Episodes with which to adapt, generated by exploration_policy exploring
        the environment.

    Returns:
      BasePolicy
    """
    old_policy, self._policy = self._policy, exploration_policy
    self._inner_algo.policy = exploration_policy
    self._inner_optimizer.module = exploration_policy

    batch_samples = self._process_samples(exploration_episodes)
    self._adapt(batch_samples, set_grad = False)

    self._policy = old_policy
    self._inner_algo.policy = self._inner_optimizer.module = old_policy

    return exploration_policy

  def _log_performance(self, itr, all_samples, loss_before, loss_after, kl_before, kl, policy_entropy):
    """
    Evaluate performance of this batch.

    Args:
      itr (int): Iteration number.
      all_samples (list[list[_MAMLEpisodeBatch]]): Two dimensional list of _MAMLEpisodeBatch of size.
      loss_before (float): Loss before optimization step.
      loss_after (float): Loss after optimization step.
      kl_before (float): KL divergence before optimization step.
      kl (float): KL divergence after optimization step.
      policy_entropy (float): Policy entropy.

    Returns:
      float: The average return in last epoch cycle.
    """
    tabular.record('Iteration', itr)

    name_map = None
    if hasattr(self._env, 'all_task_names'):
      names = self._env.all_task_names
      name_map = dict(zip(names, names))

    rtns = log_multitask_performance(
      itr,
      EpisodeBatch.from_list(
        env_spec = self._env.spec,
        paths = [
          path for task_paths in all_samples
          for path in task_paths[self._num_grad_updates].paths
        ]),
      discount = self._inner_algo.discount,
      name_map = name_map
    )

    with tabular.prefix(self._policy.unique_id + '/'):
      tabular.record('LossBefore', loss_before)
      tabular.record('LossAfter', loss_after)
      tabular.record('dLoss', loss_before - loss_after)
      tabular.record('KLBefore', kl_before)
      tabular.record('KLAfter', kl)
      tabular.record('Entropy', policy_entropy)

    return np.mean(rtns)

  @property
  def policy(self):
    """
    Current policy of the inner algorithm.

    Returns:
      BasePolicy
    """
    return self._policy

  @property
  def _old_policy(self):
    """
    Old policy of the inner algorithm.

    Returns:
      BasePolicy
    """
    return self._inner_algo._old_policy

  @property
  def exploration_policy(self):
    """
    Return a policy used before adaptation to a specific task.

    Returns:
      BasePolicy
    """
    return copy.deepcopy(self._policy)


class _MAMLEpisodeBatch(
  collections.namedtuple('_MAMLEpisodeBatch', [
      'paths', 'observations', 'actions', 'rewards', 'valids',
      'baselines'
  ])):
  """
  A tuple representing a batch of whole episodes in MAML.
  """
  pass
