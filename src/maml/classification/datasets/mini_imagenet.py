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

  def __init__(self, root_path: str, mode: str, batch_size: int, N_way: int, K_shot: int, K_query, resize: int,
               starting_idx: int = 0):
    """
    The problem of N_way-way classifcations is set up as folows: select N_way unseen classes, provide the model with
    K different instances of each of the N_way classes, and evaluate the model's ability to classify new instances within
    the N_way classes. (Reference: https://arxiv.org/abs/1703.03400)

    Meta-Learning is different from general supervised learning in terminology, and how batch and set are used.
    - A batch contains several sets.
    - A set contains N_way * K for meta-train set, and N_way * N_Query for meta-test set where N_way is the number of classes
      to meta-train on, K is the number of instances per class, and K_Query is the number of instances to meta-test.

    Args:
      root_path (str):
      mode (str): Whether the
      batch_size (int):
      N_way (int): The number of classes (N_way).
      K_shot (int): The number of instances of each of the N_way classes in training (support set).
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
    self.K = K_shot
    self.K_query = K_query

    self.set_size = self.N * self.K
    self.query_size = self.N * self.K_query

    csv_file_path = os.path.join(self.root_path, self.mode + '.csv')
    image_paths_by_label = self.load_image_paths(csv_file_path)

    self.x_support_batch = list()
    self.x_query_batch = list()

    self.data = list()
    self.label_to_images_mapping = dict()
    self.create_batch()

    for i, (label, images) in enumerate(image_paths_by_label):
      # self.data = [[image_1, image_2, ...], ..., [image_897, image_898, ...]]
      self.data.append(images)
      self.label_to_images_mapping[label] = i + self.starting_idx

    self.num_classes_total = len(self.data)
    pass

  @staticmethod
  def load_image_paths(csv_file_path: str) -> dict:
    """
    Loads the CSV file based on the file images_folder_path provided, and returns a dictionary indexed by labels and corresponding
    image files.

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

  def create_batch(self):
    """
    Create a batch of Mini-Imagenet data for the meta-learning task.

    Each batch contains multiple sets.

    Returns:
      None
    """
    self.x_support_batch = list()
    self.x_query_batch = list()

    for b in range(self.batch_size):
      # select n_way classes randomly
      selected_classes = np.random.choice(self.num_classes_total, self.N, False)
      np.random.shuffle(selected_classes)

      x_batch_support_set = list()
      x_batch_query_set = list()

      for cls in selected_classes:
        # select k_shot + K_query for each class
        selected_images_idx = np.random.choice(len(self.data[cls]), self.K + self.K_query, False)
        np.random.shuffle(selected_images_idx)

        index_support_set = np.array(selected_images_idx[:self.K])
        index_query_set = np.array(selected_images_idx[self.K:])

        """
        Get filenames of images for the current "D-train" and "D-test".
        """
        x_batch_support_set.append(np.array(self.data[cls])[index_support_set].tolist())
        x_batch_query_set.append(np.array(self.data[cls])[index_query_set].tolist())
        continue

      random.shuffle(x_batch_support_set)
      random.shuffle(x_batch_query_set)

      self.x_support_batch.append(x_batch_support_set)
      self.x_query_batch.append(x_batch_query_set)

    pass

  @property
  def transform(self):
    """
    Returns transform to be applied to images in the dataset.

    Returns:
      torchvision.transforms.Compose
    """
    if self.mode == self.MODE_MINI_IMAGENET_TRAIN:
      return transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((self.resize, self.resize)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

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

    """
    # [set_size, 3, resize, resize]
    support_x = torch.FloatTensor(self.set_size, 3, self.resize, self.resize)
    # [set_size]
    support_y = np.zeros(self.set_size, dtype = np.int32)
    # [query_size, 3, resize, resize]
    query_x = torch.FloatTensor(self.query_size, 3, self.resize, self.resize)

    # [query_size]
    query_y = np.zeros(self.query_size, dtype = np.int32)

    flatten_support_x = [os.path.join(self.images_folder_path, item)
                         for sublist in self.x_support_batch[index] for item in sublist]
    support_y = np.array(
      [self.label_to_images_mapping[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
       for sublist in self.x_support_batch[index] for item in sublist]).astype(np.int32)

    flatten_query_x = [os.path.join(self.images_folder_path, item)
                       for sublist in self.x_query_batch[index] for item in sublist]
    query_y = np.array([self.label_to_images_mapping[item[:9]]
                        for sublist in self.x_query_batch[index] for item in sublist]).astype(np.int32)

    unique_support_y = np.unique(support_y)
    random.shuffle(unique_support_y)

    support_y_relative = np.zeros(self.set_size)
    query_y_relative = np.zeros(self.query_size)

    for idx, l in enumerate(unique_support_y):
      support_y_relative[support_y == l] = idx
      query_y_relative[query_y == l] = idx

    for i, path in enumerate(flatten_support_x):
      support_x[i] = self.transform(path)

    for i, path in enumerate(flatten_query_x):
      query_x[i] = self.transform(path)

    return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

  def __len__(self):
    """
    Returns the length of the dataset.

    Returns:
      int
    """
    return self.batch_size


