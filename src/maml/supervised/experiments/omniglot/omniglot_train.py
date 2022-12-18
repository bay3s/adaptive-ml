import os
import torch
import numpy as np
import logging
import json
import time
import argparse


import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from src.maml.supervised.enums.task_type import TaskType
from src.maml.supervised.models import OmniglotCNN
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


def train(args):
  logging.basicConfig(level = logging.DEBUG)

  random_seed = 1234
  use_cuda = False
  output_folder = './output'

  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)

  # @todo check what these are and if the alternative "use_deterministic" is more appropriate to be used here.
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    logging.debug('Creating folder `{0}`'.format(output_folder))
    pass

  folder = os.path.join(output_folder, time.strftime('%Y-%m-%d_%H%M%S'))
  os.makedirs(folder)
  logging.debug('Creating folder `{0}`'.format(folder))

  args.folder = os.path.abspath(args.folder)
  args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
  outfile_path = os.path.abspath(os.path.join(folder, 'model_results.json'))

  # save the configuration in a config.json file
  with open(os.path.join(folder, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent = 2)
    pass

  logging.info('Saving configuration file in `{0}`'.format(os.path.abspath(os.path.join(folder, 'config.json'))))

  meta_train_set = omniglot(
    'data',
    ways = 5,
    shots = 5,
    test_shots = 15,
    meta_train = True,
    download = True,
    seed = random_seed
  )

  meta_train_loader = BatchMetaDataLoader(
    meta_train_set,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = args.num_workers
  )

  model = OmniglotCNN(in_channels=1, out_features = args.num_ways, hidden_size=args.hidden_size)
  meta_optim = torch.optim.Adam(model.parameters(), lr = args.meta_lr)
  loss_function = F.cross_entropy

  meta_learner = MAML(
    task_type = TaskType.CLASSIFICATION,
    model = model,
    meta_optimizer = meta_optim,
    pre_training_lr = args.pre_training_lr,
    pre_train_steps = args.num_steps,
    meta_lr = args.meta_lr,
    loss_function = loss_function,
    device = device
  )

  output = []
  _pretty_print('epoch', 'train loss', 'train acc', 'train prec', 'val loss', 'val acc', 'val prec')

  for epoch in range(args.num_epochs):
    trained_model, train_results = meta_learner.meta_train(meta_train_loader)

    _pretty_print(
      (epoch + 1),
      train_results['mean_outer_loss'],
      train_results['accuracies_after'],
      train_results['precision_after'],
    )

    output.append({
      'epoch': (epoch + 1),
      'train_loss': train_results['mean_outer_loss'],
      'train_acc': train_results['accuracies_after'],
      'train_prec': train_results['precision_after']
      # 'val_loss': val_results['mean_outer_loss'],
      # 'val_acc': val_results['accuracies_after'],
      # 'val_prec': val_results['precision_after']
    })

    if args.output_folder is not None:
      with open(outfile_path, 'w') as f:
        json.dump(output, f)
      pass

    continue

  if hasattr(meta_train_set, 'close'):
    meta_train_set.close()
    # meta_test_set.close()

  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser('MAML')

  # Debugging
  parser.add_argument('--folder', type = str, help = 'Folder where data is downloaded..')
  parser.add_argument('--output-folder', type = str, help = 'Folder to save the model.')
  parser.add_argument('--random-seed', type = int, help = 'Random seed to be used for the current experiment.')

  # K-shot settings.
  parser.add_argument('--num-ways', type = int, help = 'Number of classes per task (N in "N-way").')
  parser.add_argument('--num-shots', type = int, help = 'Number of training example per class (k in "K-shot")')
  parser.add_argument('--num-shots-test', type = int, help = 'Number of test examples per class. ')

  # Optimization
  parser.add_argument('--batch-size', type = int, help = 'Number of tasks in a batch of tasks.')
  parser.add_argument('--pre-training-steps', type = int, help = 'Number of fast pre-training steps.')
  parser.add_argument('--num-epochs', type = int, help = 'Number of epochs of meta-training.')
  parser.add_argument('--num-batches', type = int, help = 'Number of batch of tasks per epoch.')
  parser.add_argument('--meta-lr', type = float, help = 'Learning rate for the meta-meta_optimizer.')

  # Misc
  parser.add_argument('--num-workers', type = int, help = 'Number of workers to use for data-loading (default: 1).')
  parser.add_argument('--use-cuda', action = 'store_true')

  input_args = parser.parse_args()
  print(torch.device('cuda' if input_args.use_cuda and torch.cuda.is_available() else 'cpu'))

  if input_args.num_shots_test <= 0:
    input_args.num_shots_test = input_args.num_shots

  train(input_args)
  pass
