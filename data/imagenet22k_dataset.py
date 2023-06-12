import os
import json
import torch.utils.data as data
import numpy as np
import glob
from PIL import Image
import json

import warnings
import sys
sys.path.append("..")
from utils import get_rank

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class IN22KDATASET(data.Dataset):
    def __init__(self, root, folder_name, transform=None, target_transform=None):
        super(IN22KDATASET, self).__init__()

        self.data_path = os.path.join(root, folder_name)
        print("glob read file names")
        self.database = glob.glob(os.path.join(self.data_path, '**/*.JPEG'))
        print("finish read file names")
        self.transform = transform
        self.target_transform = target_transform

        self.nametocls = json.load(open("data/21kpnametocls.txt"))

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database[index]

        # images
        images = self._load_image(idb).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = self.nametocls[idb.split('/')[-2]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def __len__(self):
        return len(self.database)
