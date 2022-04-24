import os

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


"""
The code below is:
Copyright (c) 2016 Soumith Chintala
Licensed under BSD 3-Clause
(https://github.com/pytorch/vision/blob/master/LICENSE)
"""


class MVTecAD(ImageFolder):
    """
    `MVTec Anomaly Detection <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.

    Args:
        root (string): Root directory of the MVTec AD Dataset.
        dataset_name (string, optional): One of the MVTec AD Dataset names.
        train (bool, optional): If true, use the train dataset, otherwise the test dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (sample path, class_index) tuples.
        targets (list): The class_index value for each image in the dataset.

    Note:
        The normal class index is 0.
        The abnormal class indexes are assigned 1 or higher alphabetically.
    """

    dataset_urls = {'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
                    'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
                    'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
                    'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
                    'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
                    'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
                    'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
                    'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
                    'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
                    'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
                    'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
                    'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
                    'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
                    'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
                    'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz'}
    available_dataset_names = list(dataset_urls.keys())

    def __init__(self, root, dataset_name, train=True, transform=None,
                 target_transform=None, download=False):
        self.using_dataset = dataset_name.lower()
        self.train = train
        self.root = root

        if download is True:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               ' You can use download=True to download it')

        super(MVTecAD, self).__init__(self.split_folder, transform=transform,
                                      target_transform=target_transform)
        self.root = root

    @property
    def split_folder(self):
        split = 'train' if self.train is True else 'test'
        return os.path.join(self.root, self.using_dataset, split)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.split_folder))

    def download(self):
        os.makedirs(self.root, exist_ok=True)

        if self._check_exists():
            return

        if self.using_dataset not in self.available_dataset_names:
            raise ValueError('The dataset called {} is not exist.'.format(self.using_dataset))

        filename = self.using_dataset + '.tar.xz'
        url = self.dataset_urls[self.using_dataset]
        download_and_extract_archive(url, self.root, filename=filename)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove('good')
        classes = ['good'] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return sorted(classes), class_to_idx

    def extra_repr(self):
        split = 'Train' if self.train is True else 'Test'
        return 'Using Dataset: {dataset}\nSplit: {split}'.format(dataset=self.using_dataset,
                                                                 split=split)
