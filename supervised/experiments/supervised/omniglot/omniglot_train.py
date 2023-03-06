import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pathlib
from datetime import datetime

from supervised.datasets import OmniglotNShot
from supervised.learners import MAML
from supervised.enums import TaskType
from supervised.models import OmniglotCNN


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--epochs', type = int, help = 'epoch number', default = 1)
  argparser.add_argument('--n_way', type = int, help = 'n way', default = 5)
  argparser.add_argument('--k_spt', type = int, help = 'k shot for support set', default = 1)
  argparser.add_argument('--k_qry', type = int, help = 'k shot for query set', default = 10)
  argparser.add_argument('--image_size', type = int, help = 'image_size', default = 28)
  argparser.add_argument('--imgc', type = int, help = 'imgc', default = 1)
  argparser.add_argument('--task_num', type = int, help = 'meta batch size, namely task num', default = 32)
  argparser.add_argument('--meta_lr', type = float, help = 'meta-level outer learning rate', default = 1e-3)
  argparser.add_argument('--update_lr', type = float, help = 'task-level inner update learning rate', default = 0.4)
  argparser.add_argument('--update_step', type = int, help = 'task-level inner update steps', default = 5)
  argparser.add_argument('--update_step_test', type = int, help = 'update steps for finetunning', default = 10)
  argparser.add_argument('--force-download', type = bool, help = 'whether to download all the data', default = False)

  args = argparser.parse_args()
  pathlib.Path('models').mkdir(parents = True, exist_ok = True)

  print('Instantiate omniglot_n_shot.')

  device = 'cpu'

  omniglot_n_shot = OmniglotNShot(
    downloads_folder = './downloads/omniglot_n_shot',
    batch_size = args.task_num,
    N_way = args.n_way,
    K_shot = args.k_spt,
    K_query = args.k_qry,
    image_size = args.image_size,
    force_download = False
  )

  model = OmniglotCNN(
    in_channels = 1,
    hidden_size = 64,
    out_features = args.n_way
  )

  meta_learner = MAML(
    task_type = TaskType.CLASSIFICATION,
    model = model,
    meta_lr = 1e-3,
    fast_lr = 1e-3,
    fast_gradient_steps = 1,
    loss_function = torch.functional.F.cross_entropy,
    device = device
  )

  writer = SummaryWriter()

  for current_epoch in range(args.epochs):
    x_support, y_support, x_query, y_query = omniglot_n_shot.next(mode = 'train')

    model, training_losses = meta_learner.meta_train(
      x_support_batch = torch.FloatTensor(x_support).to(device),
      y_support_batch = torch.LongTensor(y_support).to(device),
      x_query_batch = torch.FloatTensor(x_query).to(device),
      y_query_batch = torch.LongTensor(y_query).to(device)
    )

    if current_epoch % 500 == 0:
      writer.add_scalar('meta-loss-omniglot-cnn', np.mean(training_losses), current_epoch)
      pass

  torch.save(model.state_dict(), f'./models/omniglot-{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}.pt')
  print('Meta-Training Successful.')
  pass

