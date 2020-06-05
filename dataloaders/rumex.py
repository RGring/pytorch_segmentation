from base import BaseDataSet, BaseDataLoader
from PIL import Image
import os
from glob import glob
from utils import palette
import numpy as np

class RumexDataset(BaseDataSet):
    """
    Custom Rumex Dataset
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RumexDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["img_train", "img_val"]:
            self.root = os.path.join(self.root, 'ds')
            self.image_dir = os.path.join(self.root, self.split)
            self.label_dir = os.path.join(self.root, 'masks_machine')
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else: raise ValueError(f"Invalid split name {self.split}")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)[:,:,0]
        return image, label, image_id

class Rumex(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.49380072, 0.59038162, 0.4732776 ]
        self.STD = [0.20804656, 0.21388439, 0.21712582]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = RumexDataset(**kwargs)
        super(Rumex, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)