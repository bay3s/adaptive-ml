import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from copy import deepcopy

import torch
import torch.nn as nn

from torchmeta.utils.data import BatchMetaDataLoader

from src.maml.supervised.enums import TaskType
from src.maml.supervised.learners.utils import (
  _tensors_to_device,
  _compute_accuracy,
  _compute_precision
)

from src.maml.supervised.models import SinusoidMLP


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
    pre_training_steps: int,
    meta_lr: float,
    pre_training_lr: float,
    loss_function: callable,
    device: str
  ):
    """
    Meta-learner class for Model-Agnostic Meta-Learning.

    Args:
      task_type (TaskType): The type of task that the model is being trained for.
      model (nn.Module): The model to be trained.
      meta_optim (torch.optim.Optimizer): The meta_optim to be used for training the model.
      meta_lr (float): Learning rate for the inner loop of MAML.
      pre_training_lr (float): Learning rate for the inner loop of MAML.
      loss_function (callable): A callable that returns the loss for the model.
      pre_training_steps (int): The number of gradient steps to be made in pre-training.

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

    self.pre_training_lr = pre_training_lr

    self.meta_lr = meta_lr
    self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

    self.loss_function = loss_function
    self.pre_training_steps = pre_training_steps
    pass

  def meta_train(self, data_loader: BatchMetaDataLoader) -> [nn.Module, list]:
    """
    Meta-train the model, and return it along with the aggregated results.

    Args:
      data_loader (BatchMetaDataLoader): The data loader for meta-training the model.

    Returns:
      MetaModule, list
    """
    self.model.train(True)
    training_losses = list()
    num_batches = 1

    for batch in data_loader:
      theta_primes = dict()

      self.meta_optim.zero_grad()
      batch = _tensors_to_device(batch, device = self.device)

      num_tasks = batch['test'].size(0) if isinstance(batch['test'], torch.Tensor) else len(batch['test'])
      meta_loss = torch.Tensor([0.], device = self.device)

      # pre-training step.
      for task_idx, (x_pre_train, y_pre_train) in enumerate(zip(*batch['train'])):
        model_copy = deepcopy(self.model)

        x_pre_train = x_pre_train.float()
        y_pre_train = y_pre_train.float()

        for step in range(self.pre_training_steps):
          y_train_predicted = model_copy(x_pre_train)
          loss = self.loss_function(y_train_predicted, y_pre_train)

          self.model.zero_grad()
          params = OrderedDict(model_copy.named_parameters())
          grads = torch.autograd.grad(loss, params.values(), create_graph = True)

          for (name, param), grad in zip(params.items(), grads):
            params[name] = param - self.pre_training_lr * grad

        # save parameters to be udpated at meta
        theta_primes[task_idx] = model_copy.state_dict()
        continue

      self.meta_optim.zero_grad()

      for task_idx, (x_test, y_test) in enumerate(zip(*batch['test'])):
        x_test = x_test.float()
        y_test = y_test.float()

        model_theta_prime = deepcopy(self.model)
        model_theta_prime.load_state_dict(theta_primes[task_idx])
        y_test_predicted = model_theta_prime(x_test)
        meta_loss += self.loss_function(y_test_predicted, y_test)
        continue

      # compute gradient and backprop based on the meta-loss.
      meta_loss.div_(num_tasks)
      meta_loss.backward()
      self.meta_optim.step()
      training_losses.append(meta_loss.item())

      num_batches += 1
      continue

    return self.model, training_losses

  @property
  def is_classification_task(self) -> bool:
    """
    Returns true if the current meta-learned model is for a classification task.

    Returns:
      bool
    """
    return self.task_type == TaskType.CLASSIFICATION
