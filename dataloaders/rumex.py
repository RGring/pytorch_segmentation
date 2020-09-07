from base import BaseDataSet, BaseDataLoader
from PIL import Image
import cv2
import os
from glob import glob
from utils import palette
import numpy as np
import math
import xml.etree.cElementTree as ET


class RumexDataset(BaseDataSet):
    """
    Custom Rumex Dataset
    """
    def __init__(self, **kwargs):
        # Must be > 3, check if set in config
        self.num_subimg_splits = 3
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RumexDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["imgs_train", "imgs_val"]:
            self.image_dir = os.path.join(self.root, self.split)
            self.annotations = self._read_cvat_annotations(os.path.join(self.root, 'ann/annotations.xml'))

            file_ids = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
            self.files = []
            for id in file_ids:
                for i in range(self.num_subimg_splits):
                    for j in range(self.num_subimg_splits):
                        self.files.append({"file_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
        else: raise ValueError(f"Invalid split name {self.split}")

    def _read_cvat_annotations(self, path_to_annotation_file):
        root = ET.parse(path_to_annotation_file).getroot()
        ann = {}
        for img in root.findall('image'):
            ann[img.attrib["name"]] = []
            for pol in img.findall("polygon"):
                points_strs = pol.attrib["points"].split(";")
                points = []
                for points_str in points_strs:
                    points_str = points_str.split(",")
                    points.append([int(float(points_str[0])), int(float(points_str[1]))])
                ann[img.attrib["name"]].append(np.array(points))
        return ann

    def _get_sub_img(self, img, split_x, split_y):
        w_img, h_img =  img.shape[0:2]
        w_subimg = math.floor(w_img / self.num_subimg_splits)
        h_subimg = math.floor(h_img / self.num_subimg_splits)
        return img[split_x * w_subimg:split_x * w_subimg + w_subimg, split_y * h_subimg:split_y * h_subimg + h_subimg, :]

    def _load_data(self, index):
        subimg_id = self.files[index]
        image_path = os.path.join(self.image_dir, subimg_id["file_id"] + '.jpg')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask_img = np.zeros(image.shape, dtype=np.int32)
        for pol in self.annotations[os.path.basename(image_path)]:
            cv2.fillPoly(mask_img, pts = [pol], color=(1, 1, 1))


        image = self._get_sub_img(image, subimg_id["split_x"], subimg_id["split_y"])
        label = (self._get_sub_img(mask_img, subimg_id["split_x"], subimg_id["split_y"]))

        # Write images for verifying correctness.
        # cropped_img = Image.fromarray(image.astype(dtype="uint8"))
        # cropped_label = np.where(label == 1, 120, label)
        # cropped_label = Image.fromarray(cropped_label.astype(dtype="uint8"))
        # mask = Image.new("L", cropped_label.size, 128)
        # out_img = Image.composite(cropped_img, cropped_label, mask)
        # id = subimg_id["sub_img_id"]
        # cropped_img.save(f"test_output/{id}.jpeg")
        # out_img.save(f"test_output/{id}_masked.jpeg")

        # For training, only one channel needed.
        label = label[:, :, 0]
        return image, label, subimg_id["sub_img_id"]

class Rumex(BaseDataLoader):
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

        self.dataset = RumexDataset(**kwargs)
        super(Rumex, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)