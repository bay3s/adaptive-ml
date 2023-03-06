import errno
import os
import zipfile

import torchvision.transforms as transforms
from six.moves import urllib

class Omniglot:
  DOWNLOAD_URL_BACKGROUND = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
  DOWNLOAD_URL_EVALUATION = 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'

  DOWNLOAD_URLS = [
    DOWNLOAD_URL_BACKGROUND,
    DOWNLOAD_URL_EVALUATION
  ]

  RAW_FOLDER = 'raw'
  PROCESSED_FOLDER = 'processed'

  def __init__(self, downloads_folder: str, input_transforms: transforms.Compose = None,
               target_transforms: transforms.Compose = None, force_download: bool = False):
    """
    Initialize the Omniglot dataset class.

    Args:
      downloads_folder (str): The downloads_folder folder for the dataset force_download.
      input_transforms (Compose): The transform to apply to the inputs for the dataset.
      target_transforms (Compose): The transform to apply to the targets of the dataset.
      force_download (bool): Whether to force_download the data.
    """
    self.root_folder = downloads_folder
    self.transform = input_transforms
    self.target_transform = target_transforms

    self._is_download_complete = None
    if force_download or not self.is_download_complete:
      self._download()

    self.all_items = self._find_classes()
    self.idx_classes = self._index_classes(self.all_items)
    pass

  def __getitem__(self, index: int):
    """
    Given an index, returns the element in the dataset at the index.

    Args:
      index (int): The index for which to return an element for.

    Returns:
      [np.ndarray, int]
    """
    filename = self.all_items[index][0]
    img = str.join('/', [self.all_items[index][2], filename])

    target = self.idx_classes[self.all_items[index][1]]
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.all_items)

  @property
  def is_download_complete(self) -> bool:
    """
    Returns true if the data has already been downloaded, Fals otherwise.

    Returns:
      bool
    """
    if self._is_download_complete is None:
      evaluation_images_path = os.path.join(self.root_folder, self.PROCESSED_FOLDER, 'images_evaluation')
      background_images_path = os.path.join(self.root_folder, self.PROCESSED_FOLDER, 'images_background')

      self._is_download_complete = os.path.exists(evaluation_images_path) and os.path.exists(background_images_path)

    return self._is_download_complete

  def _download(self) -> bool:
    """
    Download Omniglot dataset from the original source.

    Returns:
      bool
    """
    if self._is_download_complete:
      return True

    try:
      os.makedirs(os.path.join(self.root_folder, self.RAW_FOLDER))
      os.makedirs(os.path.join(self.root_folder, self.RAW_FOLDER))
    except OSError as e:
      if e.errno == errno.EEXIST:
        pass
      else:
        raise

    for download_url in self.DOWNLOAD_URLS:
      print(f'Download Omniglot data from source {download_url} ...')
      http_response = urllib.request.urlopen(download_url)
      download_file_name = download_url.rpartition('/')[2]
      download_file_path = os.path.join(self.root_folder, self.RAW_FOLDER, download_file_name)

      with open(download_file_path, 'wb') as f:
        f.write(http_response.read())

      file_processed = os.path.join(self.root_folder, self.PROCESSED_FOLDER)
      print(f'Unzip from {download_file_path} to {file_processed} ...')

      zip_file = zipfile.ZipFile(download_file_path, 'r')
      zip_file.extractall(file_processed)
      zip_file.close()

    print('Omniglot downloads complete ...')

    return True

  def _find_classes(self) -> list:
    """
    Find classes in the downloads_folder folder and return a list of said classes.

    Returns:
      list
    """
    root_folder = os.path.join(self.root_folder, self.PROCESSED_FOLDER)
    retour = list()

    for (root, _, files) in os.walk(root_folder):
      for f in files:
        if f.endswith('png'):
          r = root.split('/')
          lr = len(r)
          retour.append((f, r[lr - 2] + '/' + r[lr - 1], root))

    return retour

  @staticmethod
  def _index_classes(items: list) -> dict:
    """
    Given a list of classes, index them appropriately.

    Args:
      items (list): A list containing [?]

    Returns:

    """
    idx = dict()

    for i in items:
      if i[1] not in idx:
        idx[i[1]] = len(idx)

    return idx
