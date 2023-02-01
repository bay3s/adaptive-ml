import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pathlib

from src.supervised import SinusoidNShot
from src.supervised import MAML
from src.supervised import TaskType
from src.supervised import SinusoidMLP


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--epochs', type = int, help = 'epoch number', default = 70_000)
  argparser.add_argument('--k-spt', type = int, help = 'k shot for support set', default = 10)
  argparser.add_argument('--k-qry', type = int, help = 'k shot for query set', default = 10)
  argparser.add_argument('--image_size', type = int, help = 'image_size', default = 28)
  argparser.add_argument('--task_num', type = int, help = 'meta batch size, namely task num', default = 10)

  argparser.add_argument('--meta_lr', type = float, help = 'meta-level outer learning rate', default = 1e-3)
  argparser.add_argument('--update_lr', type = float, help = 'task-level inner update learning rate', default = 0.4)
  argparser.add_argument('--update_step', type = int, help = 'task-level inner update steps', default = 1)
  argparser.add_argument('--update_step_test', type = int, help = 'update steps for finetunning', default = 1)

  args = argparser.parse_args()
  pathlib.Path('./models').mkdir(parents = True, exist_ok = True)

  device = 'cpu'

  sinusoid_n_shot = SinusoidNShot(
    batch_size = args.task_num,
    K_shot = args.k_spt,
    K_query = args.k_qry,
    device = device
  )

  model = SinusoidMLP(input_dim = 1, hidden_dim = 40, output_dim = 1)

  meta_learner = MAML(
    task_type = TaskType.REGRESSION,
    model = model,
    meta_lr = 1e-3,
    fast_lr = 1e-3,
    fast_gradient_steps = 1,
    loss_function = torch.functional.F.mse_loss,
    device = device
  )

  writer = SummaryWriter()

  for current_epoch in range(args.epochs):
    x_support, y_support, x_query, y_query = sinusoid_n_shot.next()

    model, training_losses = meta_learner.meta_train(
      x_support_batch = torch.FloatTensor(x_support).to(device),
      y_support_batch = torch.FloatTensor(y_support).to(device),
      x_query_batch = torch.FloatTensor(x_query).to(device),
      y_query_batch = torch.FloatTensor(y_query).to(device)
    )

    writer.add_scalar('meta-loss-sinusoid', np.mean(training_losses), current_epoch)

  print(f'End of meta-training.')
  pass

