from base import BaseDataSet, BaseDataLoader
import cv2
from utils import palette
import numpy as np
from collage_generation.collage_generation import CollageGeneration

class RumexOnlineDataset(BaseDataSet):
    """
    Custom Rumex Dataset
    """
    def __init__(self, **kwargs):
        # Must be > 3, check if set in config
        self.num_subimg_splits = 0
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RumexOnlineDataset, self).__init__(**kwargs)
        self._init_img_generator()


    def _set_files(self):
        self.files = np.zeros(50)

    def _init_img_generator(self):
        background_folder = f"{self.root}/backgrounds"
        rumex_crops_folder = f"{self.root}/rumex_crops"
        ann_path = f"{self.root}/dummy"
        non_rumex_crops_folder = f"{self.root}/dummy"
        out_folder = f"{self.root}/dummy"
        self.image_generator = CollageGeneration(background_folder, rumex_crops_folder, non_rumex_crops_folder, ann_path, out_folder)

    def _write_masked_imgs(self, image, label, index):
        image = np.asarray(image, dtype=np.int32)
        label = np.where(label == 1, 120, label)
        image = cv2.addWeighted(label, 1, image, 0.8, 0)
        cv2.imwrite(f"test_output/{index}_masked.jpeg", image)

    def _load_data(self, index):
        mode = "mix4"
        success = False
        while not success:
            try:
                img_comp, polygons = self.image_generator.generate_datapoint([mode])
                success = True
            except:
                continue

        image = cv2.cvtColor(img_comp[mode], cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        mask_img = np.zeros(image.shape, dtype=np.int32)
        for polygon in polygons:
            pol = polygon.get_polygon_points_as_array()
            cv2.fillPoly(mask_img, pts = [pol], color=(1, 1, 1))

        if image.shape[0] > image.shape[1]:
            image = np.swapaxes(image, 0, 1)
            mask_img = np.swapaxes(mask_img, 0, 1)

        # Write images for verifying correctness.
        # self._write_masked_imgs(image, label, index)

        label = mask_img[:, :, 0]
        return image, label, index

class RumexOnline(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.49380072, 0.59038162, 0.4732776]
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

        self.dataset = RumexOnlineDataset(**kwargs)
        super(RumexOnline, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)