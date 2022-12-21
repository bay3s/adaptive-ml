import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from src.maml.supervised.datasets import Sinusoid
from src.maml.supervised.models import SinusoidMLP


@torch.no_grad()
def generate_task() -> Sinusoid:
  """
  Samples amplit-ude and phase, then reutrns a Sinuosid function corresponding to it.

  Returns:
    Sinusoid
  """
  amp = torch.rand(1).item() * 4.9 + 0.1
  phase = torch.rand(1).item() * np.pi

  return Sinusoid(amp, phase)


def MAML(
  num_training_epochs: int,
  num_tasks: int,
  k_train: int,
  k_test: int,
  learning_rate: float,
  device: str
):
  """
  Runs MAML on in the supervised regression setting as described in the MAML paper.

  Args:
      num_training_epochs (int): The number of epochs to run MAML over.
      num_tasks (int): Number of tasks to be sampled from the distribution.
      k_train (int): The number of samples to be used for the training set in meta-training.
      k_test (int): The number of samples to be used for the test in meta-training.
      learning_rate (float): Learning rate to be used for the optimizers.
      device (str): Device to run MAML on (can either be set to CPU or GPU)

  Returns:
      Returns a neural network that is tuned for few-shot learning.
  """

  """
  @todo check if the MAML algorithm can be further broken down, how much of this can be reused for classificaiton / reinforcement learning.
  @todo check if the learning rates for internal and meta updates should be different.
  """
  meta_model = SinusoidMLP(input_dim = 1, hidden_dim = 40, output_dim = 1).to(device)
  meta_optim = torch.optim.Adam(meta_model.parameters(), lr = learning_rate)

  writer = SummaryWriter()
  training_loss = list()
  inner_training_loops = 1

  for current_epoch in tqdm(range(num_training_epochs)):
    theta_prime = dict()
    data_prime = dict()

    """
    Sample tasks from the distribution.
    """
    regression_tasks = [generate_task() for _ in range(num_tasks)]

    """
    Iterate over each task in order to:
        - Evaluate with respect to K examples
        - Compute adapted parameters with gradient descent.
    """
    for i, task in enumerate(regression_tasks):
      # clone the model and optimize it for a specific task.
      model_copy = SinusoidMLP(input_dim = 1, hidden_dim = 40, output_dim = 1)
      model_copy.load_state_dict(meta_model.state_dict())
      model_copy.to(device)

      local_optim = torch.optim.SGD(model_copy.parameters(), lr = learning_rate)

      for _ in range(inner_training_loops):
        """
        Sample k points from a task, evaluate loss with respect to K examples.
        """
        x_batch, y_batch = task.sample(k_train)
        loss_function = nn.MSELoss()
        loss = loss_function(model_copy(x_batch.to(device)), y_batch.to(device))

        """
        Compute adapted parameters with gradient descent.
        """
        local_optim.zero_grad()
        loss.backward()
        local_optim.step()
        continue

      """
      For the meta-update, we need to:
          - Track adapted parameter "theta prime".
          - Sample additional data from the current task that will be used to check out-of-sample performance.
      """
      theta_prime[i] = model_copy.state_dict()

      # @todo check if this update makes a large difference.
      x_prime_batch, y_prime_batch = task.sample(k_test)
      data_prime[i] = (x_prime_batch, y_prime_batch)
      pass

    """
    While making the meta-update:
        - Sample data points for each task.
        - Compute loss for each of the "theta prime" parameters.
        - Aggregate gradients for losses corresponding to each "theta prime".
    """
    meta_optim.zero_grad()
    meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
    batch_training_loss = list()

    for i, _ in enumerate(regression_tasks):
      x_prime_batch, y_prime_batch = data_prime[i]

      model_theta_prime = SinusoidMLP(input_dim = 1, hidden_dim = 40, output_dim = 1).to(device)
      model_theta_prime.load_state_dict(theta_prime[i])
      loss_function = nn.MSELoss()

      """
      Compute out-of-sample loss for each of the tasks.
      """
      loss = loss_function(model_theta_prime(x_prime_batch.to(device)), y_prime_batch.to(device))
      loss.backward()

      batch_training_loss.append(loss.item())

      """
      Then, aggregate the gradients.
      """
      for grad, param in zip(meta_grads, model_theta_prime.parameters()):
        grad += param.grad

    """
    Update parameter updates using aggregated gradients.
    """
    for param, grad in zip(meta_model.parameters(), meta_grads):
      param.grad = grad
      pass

    meta_optim.step()
    training_loss.append(np.mean(batch_training_loss))
    writer.add_scalar('meta-loss-sinusoid', np.mean(batch_training_loss), current_epoch)

    if current_epoch % 500 == 0:
      writer.flush()

  writer.close()

  return meta_model, training_loss


def k_shot_tune(model: nn.Module, task, k_shot: int, gradient_steps, alpha, device = 'cpu'):
  optimizer = torch.optim.SGD(model.parameters(), lr = alpha)
  x_batch, target = task.sample(k_shot)

  for epoch in range(gradient_steps):
    loss_fct = nn.MSELoss()

    loss = loss_fct(model(x_batch.to(device)), target.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return model


if __name__ == '__main__':
  device = 'cpu'
  plt.rcParams['figure.figsize'] = (10, 4)

  model, training_loss = MAML(
      num_training_epochs=70000,
      k_train=10,
      k_test=25,
      learning_rate=1e-3,
      device=device,
      num_tasks=10
  )

  plt.plot(training_loss)

  x = torch.linspace(-5, 5, 50)
  task = generate_task()
  ground_truth_y = task._amplitude * torch.sin(x + task._phase)
  pre_tuning_y = model(x[..., None])

  # tune the model
  tuned_model = k_shot_tune(deepcopy(model), task, k_shot = 10, gradient_steps = 10, alpha = 1e-3, device = 'cpu')
  y = tuned_model(x[..., None])

  # plot
  plt.title('MAML, K=10')
  plt.plot(x.data.numpy(), ground_truth_y.data.numpy(), c = 'red', label = 'Ground Truth')
  plt.plot(x.data.numpy(), pre_tuning_y.data.numpy(), c = 'gray', linestyle = 'dotted', label = 'Pre-Tuning')
  plt.plot(x.data.numpy(), y.data.numpy(), c = 'mediumseagreen', label = '10 Gradient Steps', linewidth = '2')
  plt.legend()
  pass
