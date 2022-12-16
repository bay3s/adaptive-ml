import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

import csv
import random


class MiniImagenet(Dataset):

  MODE_MINI_IMAGENET_TRAIN = 'train'
  MODE_MINI_IMAGENET_VALIDATION = 'validation'
  MODE_MINI_IMAGENET_TEST = 'test'

  def __init__(self, root_path: str, mode: str, batch_size: int, N_way: int, K_support: int, K_query, resize: int,
               starting_idx: int = 0):
    """
    The problem of N_way-way classifcations is set up as folows: select N_way unseen classes, provide the model with
    K_support different instances of each of the N_way classes, and evaluate the model's ability to classify new instances within
    the N_way classes. (Reference: https://arxiv.org/abs/1703.03400)

    Meta-Learning is different from general supervised learning in terminology, and how batch and set are used.

    - A batch contains several sets that may be thought of as "tasks".
    - A set contains N_way * K_support for meta-train set, and N_way * N_Query for meta-test set where N_way is the number of
      classes to meta-train on, K_support is the number of instances per class, and K_Query is the number of instances to
      meta-test.

    Another nuance is the notion of support and query sets in meta-learning:
    - A "support" is used for training and in a K_support-shot setting would contain K_support examples used for training / fine-tuning.
    - A "query" set is used for testing the results of the meta-training process.

    Args:
      root_path (str):
      mode (str): Whether the
      batch_size (int):
      N_way (int): The number of classes (N_way).
      K_support (int): The number of instances of each of the N_way classes in training (support set).
      K_query (int): The number of instances of each of the N_way classes in evaluation (query set).
      resize (int): The dimensions to resize the images to.
      starting_idx (int): Start to index labels from.
    """

    self.root_path = root_path
    self.mode = mode
    self.starting_idx = starting_idx
    self.images_folder_path = os.path.join(self.root_path, 'images')
    self.batch_size = batch_size
    self.resize = resize

    self.N = N_way
    self.K_support = K_support
    self.K_query = K_query

    self.support_set_size = self.N * self.K_support
    self.query_set_size = self.N * self.K_query

    csv_file_path = os.path.join(self.root_path, self.mode + '.csv')
    # image_paths_by_label = [1 => [image_1, image_2, ...], ..., 89 => [image_897, image_898, ...]]
    image_paths_by_label = self.load_image_paths(csv_file_path)

    self.data = list()
    self.label_to_images_mapping = dict()

    for i, (label, images) in enumerate(image_paths_by_label):
      # self.data = [[image_1, image_2, ...], ..., [image_897, image_898, ...]]
      self.data.append(images)
      self.label_to_images_mapping[label] = i + self.starting_idx
      continue

    self.num_classes_total = len(self.data)
    self.support_batch_x, self.query_batch_x = self.create_batches()
    pass

  @staticmethod
  def load_image_paths(csv_file_path: str) -> dict:
    """
    Loads the CSV file based on the file images_folder_path provided, and returns a dictionary indexed by labels and
    corresponding image files.

    Args:
      csv_file_path (str): CSV file images_folder_path.

    Returns:
      dict
    """
    images_by_label = dict()

    with open(csv_file_path) as csv_file:
      csv_file_reader = csv.reader(csv_file, delimeter=',')

      _ = next(csv_file_reader)

      for i, row in enumerate(csv_file_reader):
        file_name = row[0]
        label = row[1]

        if label in images_by_label.keys():
          images_by_label[label].append(file_name)
        else:
          images_by_label[label] = list([file_name])
        continue

    return images_by_label

  def create_batches(self) -> [list, list]:
    """
    Create a batch of Mini-Imagenet data for the meta-learning task.

    The support and query batches both contain multiple sets of images determined based on the .

    Returns:
      [list, list]
    """
    suport_batch_x = list()
    query_batch_x = list()

    for b in range(self.batch_size):
      # select n_way classes randomly
      selected_classes = np.random.choice(self.num_classes_total, self.N, False)
      np.random.shuffle(selected_classes)

      support_set_x = list()
      query_set_x = list()

      for cls in selected_classes:
        # select k_shot + K_query for each class
        selected_image_indices = np.random.choice(len(self.data[cls]), self.K_support + self.K_query, False)
        np.random.shuffle(selected_image_indices)

        index_support_set = np.array(selected_image_indices[:self.K_support])
        index_query_set = np.array(selected_image_indices[self.K_support:])

        """
        Get filenames of images for the support and query sets ("d-train" and "d-test").
        """
        support_set_x.append(np.array(self.data[cls])[index_support_set].tolist())
        query_set_x.append(np.array(self.data[cls])[index_query_set].tolist())
        continue

      """
      Randomize and append the support and query sets to respective batches.
      """
      random.shuffle(support_set_x)
      random.shuffle(query_set_x)

      suport_batch_x.append(support_set_x)
      query_batch_x.append(query_set_x)
      continue

    return suport_batch_x, query_batch_x

  @property
  def transform(self):
    """
    Returns transform to be applied to images in the dataset.

    Returns:
      torchvision.augmentations.Compose
    """
    return transforms.Compose([
      lambda x: Image.open(x).convert('RGB'),
      transforms.Resize((self.resize, self.resize)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

  def __getitem__(self, index: int):
    """
    Returns an item from the Mini-Imagenet dataset.

    Args:
      index (int): This referes to the index of sets, 0 <= index <= batch_size - 1

    Returns:
      [torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]
    """
    support_x = torch.FloatTensor(self.support_set_size, 3, self.resize, self.resize)
    flatten_support_x = [
      os.path.join(self.images_folder_path, item) for sublist in self.support_batch_x[index] for item in sublist
    ]

    for i, path in enumerate(flatten_support_x):
      support_x[i] = self.transform(path)
      continue

    support_y = np.array([
      # filename:n0153282900000005.jpg, the first 9 characters treated as label
      self.label_to_images_mapping[item[:9]] for sublist in self.support_batch_x[index] for item in sublist
    ]).astype(np.int32)

    support_y_unique = np.unique(support_y)
    random.shuffle(support_y_unique)

    query_x = torch.FloatTensor(self.query_set_size, 3, self.resize, self.resize)
    flatten_query_x = [os.path.join(self.images_folder_path, item) for sublist in self.query_batch_x[index] for item in
                       sublist]
    for i, path in enumerate(flatten_query_x):
      query_x[i] = self.transform(path)

    query_y = np.array(
      # filename:n0153282900000005.jpg, the first 9 characters treated as label
      [self.label_to_images_mapping[item[:9]] for sublist in self.query_batch_x[index] for item in sublist]
    ).astype(np.int32)

    support_y_relative = np.zeros(self.support_set_size)
    query_y_relative = np.zeros(self.query_set_size)

    for idx, l in enumerate(support_y_unique):
      support_y_relative[support_y == l] = idx
      query_y_relative[query_y == l] = idx

    return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

  def __len__(self):
    """
    Returns the length of the dataset.

    Returns:
      int
    """
    return self.batch_size


