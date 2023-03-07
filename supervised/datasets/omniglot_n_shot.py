import gc
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from supervised.datasets import Omniglot


class OmniglotNShot:
  DATASET_FILE_NPY = 'omniglot_n_shot.npy'

  def __init__(self, downloads_folder: str, batch_size: int, N_way: int, K_shot: int, K_query: int, image_size: int,
               force_download: bool):
    """
    Initialize the Omniglot N-shot dataset.

    Args:
      downloads_folder (str): The folder where to force_download the dataset.
      batch_size (int): Batch size for each of the
      N_way (int): N-ways for meta-learning.
      K_shot (int): K-shot for the training step in meta-training.
      K_query (int): K-query for the testing step in meta-training.
      image_size (int): The image size for the dataset.
      force_download (bool): Whether to force_download the data, if this is set to True then the data is downloaded
      regardless of whether it already exists.
    """
    self.downloads_folder = downloads_folder

    self.N_way = N_way

    self.K_shot = K_shot
    self.K_query = K_query

    assert K_shot + K_query <= 20
    self.resize = image_size
    self.batch_size = batch_size

    self.x = self._load_dataset(force_download)
    self.x_train, self.x_test = self.x[:1200], self.x[1200:]
    self.N_cls = self.x.shape[0]

    # save pointer of current read batch in total cache
    self.indexes = {
      'train': 0,
      'test': 0
    }

    self.datasets = {
      'train': self.x_train,
      'test': self.x_test
    }

    self.datasets_cache = {
      'train': self._load_batches(self.datasets['train']),
      'test': self._load_batches(self.datasets['test'])
    }

    pass

  def _load_dataset(self, force_download: bool) -> np.array:
    """
    Download and instantiate the Omniglot N-shot dataset.

    Args:
      force_download (bool): Whether to force the download.

    Returns:
      np.array
    """
    if not os.path.isfile(os.path.join(self.downloads_folder, self.DATASET_FILE_NPY)) or force_download:
      image_transforms = transforms.Compose(
        [
          lambda x: Image.open(x).convert('L'),
          lambda x: x.resize((self.resize, self.resize)),
          lambda x: np.reshape(x, (self.resize, self.resize, 1)),
          lambda x: np.transpose(x, [2, 0, 1]),
          lambda x: x / 255.
        ]
      )

      x = Omniglot(
        downloads_folder = self.downloads_folder,
        input_transforms = image_transforms,
        force_download = force_download
      )

      temp = dict()

      for (img, label) in x:
        if label in temp.keys():
          temp[label].append(img)
        else:
          temp[label] = [img]

      x = list()
      for label, imgs in temp.items():
        x.append(np.array(imgs))

      del temp
      gc.collect()

      x = np.array(x).astype(np.float32)
      np.save(os.path.join(self.downloads_folder, self.DATASET_FILE_NPY), x)
    else:
      x = np.load(os.path.join(self.downloads_folder, self.DATASET_FILE_NPY))

    return x

  def _load_batches(self, data_pack):
    """
    Collects several batches data for N-shot learning.

    :param data_pack: [cls_num, 20, 84, 84, 1]

    :return:
      A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    support_set_size = self.K_shot * self.N_way
    query_set_size = self.K_query * self.N_way
    batches = []

    # print('preload next 50 caches of batch_size of batch.')
    for sample in range(10):  # num of episodes

      x_supports, y_supports, x_queries, y_queries = [], [], [], []

      for i in range(self.batch_size):
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        selected_cls = np.random.choice(data_pack.shape[0], self.N_way, False)

        for j, cur_class in enumerate(selected_cls):
          selected_img = np.random.choice(20, self.K_shot + self.K_query, False)

          # meta-training and meta-test
          x_spt.append(data_pack[cur_class][selected_img[:self.K_shot]])
          x_qry.append(data_pack[cur_class][selected_img[self.K_shot:]])
          y_spt.append([j for _ in range(self.K_shot)])
          y_qry.append([j for _ in range(self.K_query)])
          pass

        # shuffle inside a batch
        perm = np.random.permutation(self.N_way * self.K_shot)
        x_spt = np.array(x_spt).reshape(self.N_way * self.K_shot, 1, self.resize, self.resize)[perm]
        y_spt = np.array(y_spt).reshape(self.N_way * self.K_shot)[perm]
        perm = np.random.permutation(self.N_way * self.K_query)
        x_qry = np.array(x_qry).reshape(self.N_way * self.K_query, 1, self.resize, self.resize)[perm]
        y_qry = np.array(y_qry).reshape(self.N_way * self.K_query)[perm]

        # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
        x_supports.append(x_spt)
        y_supports.append(y_spt)
        x_queries.append(x_qry)
        y_queries.append(y_qry)

      # [b, setsz, 1, 84, 84]
      x_supports = np.array(x_supports).astype(np.float32).reshape(self.batch_size, support_set_size, 1, self.resize,
                                                                   self.resize)
      y_supports = np.array(y_supports).astype(np.int32).reshape(self.batch_size, support_set_size)
      # [b, qrysz, 1, 84, 84]
      x_queries = np.array(x_queries).astype(np.float32).reshape(self.batch_size, query_set_size, 1, self.resize,
                                                                 self.resize)
      y_queries = np.array(y_queries).astype(np.int32).reshape(self.batch_size, query_set_size)

      batches.append([x_supports, y_supports, x_queries, y_queries])

    return batches

  def next(self, mode: str):
    """
    Gets next batch from the dataset with name.

    Args:
      mode (str): The name of the data split (one of "train", "val", "test")

    Returns:

    """
    if self.indexes[mode] >= len(self.datasets_cache[mode]):
      self.indexes[mode] = 0
      self.datasets_cache[mode] = self._load_batches(self.datasets[mode])
      pass

    next_batch = self.datasets_cache[mode][self.indexes[mode]]
    self.indexes[mode] += 1

    return next_batch
