import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

import torch
import torch.nn as nn

from src.supervised.enums import TaskType


@dataclass
class _batch_metrics:

  num_tasks: int
  pre_training_steps: int
  is_classification_task: bool

  prior_accuracy: np.array = field(init = False)
  prior_precision: np.array = field(init = False)

  test_accuracy: np.array = field(init = False)
  test_precision: np.array = field(init = False)

  def __post_init__(self):
    self.pre_training_losses = np.zeros((self.pre_training_steps, self.num_tasks), dtype = np.float32)
    self.test_losses = np.zeros((self.num_tasks,), dtype = np.float32)

    if self.is_classification_task:
      self.prior_accuracy = np.zeros((self.num_tasks,), dtype = np.float32)
      self.prior_precision = np.zeros((self.num_tasks,), dtype = np.float32)

      self.test_accuracy = np.zeros((self.num_tasks,), dtype = np.float32)
      self.test_precisions = np.zeros((self.num_tasks,), dtype = np.float32)
      pass


class MAML:

  def __init__(
    self,
    task_type: TaskType,
    model: nn.Module,
    meta_lr: float,
    fast_lr: float,
    fast_gradient_steps: int,
    loss_function: callable,
    device: str
  ):
    """
    Meta-learner class for Model-Agnostic Meta-Learning.

    Args:
      task_type (TaskType): The type of task that the model is being trained for.
      model (nn.Module): The model to be trained.
      meta_lr (float): Learning rate for the meta optimizer.
      fast_lr (int): Learning rate for the inner loop of MAML.
      fast_gradient_steps (int): Number of gradients steps for the "fast weights".
      loss_function (callable): A callable that returns the loss for the model.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)
    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    self.task_type = task_type
    self.device = device

    self.model = model.to(device=torch.device(device))

    self.meta_lr = meta_lr
    self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

    self.fast_lr = fast_lr
    self.fast_gradient_steps = fast_gradient_steps

    self.loss_function = loss_function
    pass

  def meta_train(self, x_support_batch: torch.Tensor, y_support_batch: torch.Tensor, x_query_batch: torch.Tensor,
                 y_query_batch: torch.Tensor) -> [nn.Module, list]:
    """
    Meta-train the model, and return it along with key metrics associated with the meta-training process.

    Args:
      x_support_batch (torch.Tensor): Inputs for the support set.
      y_support_batch (torch.Tensor): Expected outputs for the support set.
      x_query_batch (torch.Tensor): Inputs for the query set.
      y_query_batch (torch.Tensor): Expected outputs for the query set.

    Returns:
      list
    """
    num_tasks = x_support_batch.shape[0]
    self.meta_optim.zero_grad()

    theta_primes = dict()

    for i in range(num_tasks):
      # set up the training set for the current batch.
      x_support_current, y_support_current = x_support_batch[i], y_support_batch[i]

      # clone the shared model before updating "fast weights".
      model_fast = deepcopy(self.model)
      model_fast.to(self.device)

      # @todo ideally don't want to create a new optimizer for each training batch
      fast_optim = torch.optim.SGD(model_fast.parameters(), lr = self.fast_lr)

      for _ in range(self.fast_gradient_steps):
        # sample k points from a task, evaluate loss with respect to K examples.
        fast_loss = self.loss_function(model_fast(x_support_current), y_support_current)

        # compute adapted parameters with gradient descent.
        fast_optim.zero_grad()
        fast_loss.backward()
        fast_optim.step()
        continue

      # keep track of the parameters
      theta_primes[i] = model_fast.state_dict()
      pass

    self.meta_optim.zero_grad()
    meta_grads = [torch.zeros_like(p) for p in self.model.parameters()]

    # instead of doing this, start logging the output to tensorboard
    batch_training_loss = list()

    for i in range(num_tasks):
      theta_prime_current = theta_primes[i]
      x_query_current, y_query_current = x_query_batch[i], y_query_batch[i]

      model_theta_prime = deepcopy(self.model)
      model_theta_prime.load_state_dict(theta_prime_current)
      y_predicted = model_theta_prime(x_query_current)

      loss = self.loss_function(y_predicted, y_query_current)
      loss.backward()

      batch_training_loss.append(loss.item())

      # aggregate the gradients.
      for grad, param in zip(meta_grads, model_theta_prime.parameters()):
        grad += param.grad

    # update parameters with the aggregrated gradients.
    for param, grad in zip(self.model.parameters(), meta_grads):
        param.grad = grad
        pass

    # update parameters based on aggregate gradients.
    self.meta_optim.step()

    return self.model, batch_training_loss

  def k_shot_tune(self):
    pass

  @property
  def is_classification_task(self) -> bool:
    """
    Returns true if the current meta-learned model is for a classification task.

    Returns:
      bool
    """
    return self.task_type == TaskType.CLASSIFICATION
