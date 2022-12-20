import os
import torch
import numpy as np
import logging
import json
import time

from dataclasses import dataclass
import dataclasses

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

from src.maml.supervised.enums.task_type import TaskType
from src.maml.supervised.models import SinusoidMLP
from src.maml.supervised.learners import MAML


def _format_val(v, col_width: int):
  if isinstance(v, float):
    v = "{:.{}f}".format(v, 5)

  if isinstance(v, int):
    v = str(v)

  return v.ljust(col_width)


def _pretty_print(*values):
  col_width = 13
  str_values = [_format_val(v, col_width) for v in values]
  print("   ".join(str_values))
  pass


@dataclass
class _input_args:

  num_epochs: int

  meta_lr: float
  pre_training_lr: float
  pre_training_steps: int

  num_ways: int
  num_shots_train: int
  num_shots_test: int

  use_cuda: bool
  random_seed: int
  output_folder: str = './output'

  pass


def train(args: _input_args):
  logging.basicConfig(level = logging.DEBUG)

  np.random.seed(args.random_seed)
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)

  device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    logging.debug('Creating folder `{0}`'.format(args.output_folder))
    pass

  output_folder_path = os.path.join(args.output_folder, time.strftime('%Y-%m-%d_%H%M%S'))
  os.makedirs(output_folder_path)
  logging.debug('Creating folder `{0}`'.format(output_folder_path))

  args.model_path = os.path.abspath(os.path.join(output_folder_path, 'model.th'))
  outfile_path = os.path.abspath(os.path.join(output_folder_path, 'model_results.json'))

  # save the configuration in a config.json file
  with open(os.path.join(output_folder_path, 'config.json'), 'w') as f:
    json.dump(dataclasses.asdict(args), fp=f)
    pass

  logging.info('Saving configuration file in `{0}`'.format(os.path.abspath(os.path.join(output_folder_path, 'config'
                                                                                                            '.json'))))
  model = SinusoidMLP(input_dim=1, hidden_dim=40, output_dim=1).to(device)
  loss_function = F.mse_loss

  meta_learner = MAML(
    task_type = TaskType.REGRESSION,
    model = model,
    pre_training_lr = args.pre_training_lr,
    meta_lr = args.meta_lr,
    pre_training_steps = args.pre_training_steps,
    loss_function = loss_function,
    device = device
  )

  meta_train_set = sinusoid(shots = args.num_shots_train, test_shots = args.num_shots_test, seed = args.random_seed,
                            num_tasks = 23)

  # make sure this is working for different batch sizes.
  meta_train_loader = BatchMetaDataLoader(meta_train_set, shuffle = True)

  output = []
  _pretty_print('epoch', 'train loss', 'train acc', 'train prec', 'val loss', 'val acc', 'val prec')
  writer = SummaryWriter()

  for current_epoch in range(args.num_epochs):
    model, batch_losses = meta_learner.meta_train(meta_train_loader)
    writer.add_scalar('meta-loss-sinusoid', np.mean(batch_losses), current_epoch)

    if current_epoch % 1000 == 0:
      writer.flush()
    #
    # _pretty_print(
    #   (epoch + 1),
    #   train_results['mean_outer_loss'],
    #   train_results['accuracies_after'],
    #   train_results['precision_after'],
    # )
    #
    # output.append({
    #   'epoch': (epoch + 1),
    #   'train_loss': train_results['mean_outer_loss'],
    #   'train_acc': train_results['accuracies_after'],
    #   'train_prec': train_results['precision_after']
    #   # 'validation_loss': val_results['mean_outer_loss'],
    #   # 'validation_accuracy': val_results['accuracies_after'],
    #   # 'validation_precision': val_results['precision_after']
    # })

    # if args.output_folder is not None:
    #   with open(outfile_path, 'w') as f:
    #     json.dump(output, f)
    #   pass
    #
    # continue

  if hasattr(meta_train_set, 'close'):
    meta_train_set.close()
    # meta_test_set.close()

  pass


_training_configs = {
  'meta_lr': 1e-3,
  'num_epochs': 70_000,
  'pre_training_lr': 1e-3,
  'pre_training_steps': 1,
  'num_ways': 10,
  'num_shots_train': 10,
  'num_shots_test': 10,
  'use_cuda': False,
  'random_seed': 123,
  'output_folder': './output'
}

# @todo check if there is an alternative to using these boolean flags.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train(_input_args(**_training_configs))

pass
