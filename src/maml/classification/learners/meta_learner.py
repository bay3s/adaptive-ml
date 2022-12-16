import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import Union
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss

from src.maml.classification.modules.meta_module import MetaModule
from .utils import _tensors_to_device, _compute_accuracy, _compute_precision
from .task_type import TaskType


class MAML:

  def __init__(self, task_type: TaskType, model: MetaModule, optimizer: torch.optim.Optimizer, learning_rate: float,
               loss_function: Union[CrossEntropyLoss, MSELoss], adaptation_steps: int,
               device: str):
    """
    Meta-learner class for Model-Agnostic Meta-Learning.

    Args:
      task_type (TaskType): The type of task that the model is being trained for.
      model (nn.Module): The model to be trained.
      optimizer (torch.optim.Optimizer): The optimizer to be used for training the model.
      learning_rate (float): Learning rate for the inner loop of MAML.
      loss_function (_Loss): A callable that returns the loss for the model.
      adaptation_steps (int): The number of adaptation steps

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

    self.optimizer = optimizer
    self.learning_rate = learning_rate

    self.loss_function = loss_function
    self.adaptation_steps = adaptation_steps
    pass

  def train(self, data_loader, max_batches: int, verbose: bool = True, **kwargs):
    """

    Args:
      data_loader ():
      max_batches ():
      verbose ():
      **kwargs ():

    Returns:

    """
    mean_outer_loss, mean_accuracy, mean_precision, num_iterations = 0., 0., 0., 1.

    for results in self.training_iterations(data_loader, max_batches):
      mean_outer_loss += (results['mean_outer_loss'] - mean_outer_loss) / num_iterations

      postfix = {
        'loss': '{0:.4f}'.format(results['mean_outer_loss'])
      }

      if 'accuracies_after' in results:
        mean_accuracy += (np.mean(results['accuracies_after']) - mean_accuracy) / num_iterations
        mean_precision += (np.mean(results['precision_after']) - mean_precision) / num_iterations
        pass

      num_iterations += 1
      continue

    mean_results = {'mean_outer_loss': mean_outer_loss}

    return mean_results
    pass

  def get_outer_loss(self, batch) -> [float, dict]:
    if 'test' not in batch:
      raise RuntimeError('Batch does not contain any test dataset.')

    is_classification = self.task_type == TaskType.CLASSIFICATION

    _, test_targets = batch['test']
    num_tasks = test_targets.size(0)

    results = {
      'num_tasks': num_tasks,
      'inner_losses': np.zeros((self.adaptation_steps, num_tasks), dtype=np.float32),
      'outer_losses': np.zeros((num_tasks, ), dtype=np.float32),
      'mean_outer_loss': 0.
    }

    if is_classification:
      results.update({
        'accuracies_before': np.zeros((num_tasks,), dtype = np.float32),
        'accuracies_after': np.zeros((num_tasks,), dtype = np.float32),
        'precision_before': np.zeros((num_tasks,), dtype = np.float32),
        'precision_after': np.zeros((num_tasks,), dtype = np.float32)
      })

    mean_outer_loss = torch.Tensor(0., device=self.device)

    for task_id, (x_train, y_train, x_test, y_test) in enumerate(zip(*batch['train'], *batch['test'])):
      params, adaptation_results = self.adapt(x_train, y_train)
      results['inner_losses'][:, task_id] = adaptation_results['inner_losses']

      if is_classification:
        results['accuracies_before'][task_id] = adaptation_results['accuracy_before']
        results['precision_before'][task_id] = adaptation_results['precision_before']

      with torch.set_grad_enabled(self.model.training):
        y_test_predicted = self.model(x_test, params=params)
        outer_loss = self.loss_function(y_test_predicted, y_test)
        results['outer_losses'][task_id] = outer_loss.item()
        pass

      if is_classification:
        results['accuracies_after'][task_id] = _compute_accuracy(y_test_predicted, y_test)
        results['precision_after'][task_id] = _compute_precision(y_test_predicted, y_test)

    mean_outer_loss.div_(num_tasks)
    results['mean_outer_loss'] = mean_outer_loss.item()

    return mean_outer_loss, results

  def adapt(self, x_train: torch.Tensor, y_train: torch.Tensor):
    is_classification = self.task_type == TaskType.CLASSIFICATION

    params = None
    results = {
      'inner_losses': np.zeros((self.adaptation_steps,), dtype=np.float32)
    }

    for step in range(self.adaptation_steps):
      y_train_predicted = self.model(x_train, params=params)
      adaptation_loss = self.loss_function(y_train_predicted, y_train)

      results['inner_losses'][step] = adaptation_loss.item()

      if step == 0 and is_classification:
        results['accuracy_before'] = _compute_accuracy(y_train_predicted, y_train)
        results['precision_before'] = _compute_precision(y_train_predicted, y_train)

      self.model.zero_grad()

      params = OrderedDict(self.model.named_parameters())
      grads = torch.autograd.grad(adaptation_loss, params.values(), create_graph=False)

      for (name, param), grad in zip(params.items(), grads):
        params[name] = param - self.learning_rate * grad
        continue

    return params, results

  def training_iterations(self, data_loader, max_batches: int) -> Iterable:
    num_batches = 0

    for batch in data_loader:
      if num_batches >= max_batches:
        break

      self.optimizer.zero_grad()
      batch = _tensors_to_device(batch, device=self.device)
      outer_loss, results = self.get_outer_loss(batch)

      yield results

      outer_loss.backward()
      self.optimizer.step()
      num_batches += 1

      pass
