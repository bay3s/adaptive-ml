import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torchmeta.utils.data import BatchMetaDataLoader

from src.maml.supervised.models.modules import MetaModule
from src.maml.supervised.enums import TaskType
from src.maml.supervised.learners.utils import (
  _tensors_to_device,
  _compute_accuracy,
  _compute_precision
)


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
    model: MetaModule,
    pre_train_steps: int,
    meta_optimizer: torch.optim.Optimizer,
    meta_lr: float,
    pre_training_lr: float,
    loss_function: callable,
    device: str
  ):
    """
    Meta-learner class for Model-Agnostic Meta-Learning.

    Args:
      task_type (TaskType): The type of task that the model is being trained for.
      model (MetaModule): The model to be trained.
      meta_optimizer (torch.optim.Optimizer): The meta_optimizer to be used for training the model.
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

    self.meta_optimizer = meta_optimizer
    self.meta_lr = meta_lr
    self.pre_training_lr = pre_training_lr

    self.loss_function = loss_function
    self.pre_training_steps = pre_train_steps
    pass

  def meta_train(self, data_loader: BatchMetaDataLoader) -> [MetaModule, list]:
    """
    Meta-train the model, and return it along with the aggregated results.

    Args:
      data_loader (BatchMetaDataLoader): The data loader for meta-training the model.

    Returns:
      MetaModule, list
    """
    batch_metrics_all = list()

    for batch in data_loader:
      if 'test' not in batch:
        raise RuntimeError('Batch does not contain any test dataset.')

      self.meta_optimizer.zero_grad()
      batch = _tensors_to_device(batch, device=self.device)

      num_tasks = batch['test'].size(0)
      meta_loss = torch.Tensor(0., device = self.device)

      batch_metrics_current = _batch_metrics(num_tasks, self.pre_training_steps, self.is_classification_task)

      for task_id, (x_pre_train, y_pre_train, x_test, y_test) in enumerate(zip(*batch['train'], *batch['test'])):
        params, pre_training_losses, prior_accuracy, prior_precision = self._pre_train(x_pre_train, y_pre_train)

        with torch.set_grad_enabled(self.model.training):
          y_test_predicted = self.model(x_test, params = params)
          loss = self.loss_function(y_test_predicted, y_test)
          meta_loss += loss
          pass

        # update the results for the current batch, make this a function in the batch metrics.
        batch_metrics_current.pre_training_losses[:, task_id] = pre_training_losses
        batch_metrics_current.test_losses[task_id] = loss.item()

        if self.is_classification_task:
          batch_metrics_current.prior_accuracy[task_id] = prior_accuracy
          batch_metrics_current.prior_precision[task_id] = prior_precision
          batch_metrics_current.test_accuracy[task_id] = _compute_accuracy(y_test_predicted, y_test)
          batch_metrics_current.test_precision[task_id] = _compute_precision(y_test_predicted, y_test)
          pass

        continue

      meta_loss.div_(num_tasks)
      meta_loss.backward()
      self.meta_optimizer.step()

      batch_metrics_all.append(batch_metrics_current)
      continue

    return self.model, batch_metrics_all

  def _pre_train(self, x_train: torch.Tensor, y_train: torch.Tensor) -> (float, float, np.array, OrderedDict):
    """
    Runs the pre-training step of MAML.

    Args:
      x_train (torch.Tensor): Tensor containing inputs for pre-training.
      y_train (torch.Tensor): Tensor containing expected outputs for pre-training.

    Returns:
      (float, float, np.array, OrderedDict)
    """
    pre_training_losses = np.zeros((self.pre_training_steps,), dtype = np.float32)
    params = None
    prior_accuracy, prior_precision = 0., 0.

    for step in range(self.pre_training_steps):
      y_train_predicted = self.model(x_train, params=params)
      loss = self.loss_function(y_train_predicted, y_train)

      if step == 0:
        prior_accuracy = _compute_accuracy(y_train_predicted, y_train)
        prior_precision = _compute_precision(y_train_predicted, y_train)
        pass

      self.model.zero_grad()
      params = OrderedDict(self.model.named_parameters())
      grads = torch.autograd.grad(loss, params.values(), create_graph= self.model.training)

      for (name, param), grad in zip(params.items(), grads):
        params[name] = param - self.pre_training_lr * grad
        continue

    return params, pre_training_losses, prior_accuracy, prior_precision

  @property
  def is_classification_task(self) -> bool:
    """
    Returns true if the current meta-learned model is for a classification task.

    Returns:
      bool
    """
    return self.task_type == TaskType.CLASSIFICATION
