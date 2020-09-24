from base import BaseDataSet, BaseDataLoader
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
        self.num_subimg_splits = 0
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RumexDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.annotations = {}
        self.files = []
        for split in self.split:
            image_dir = os.path.join(self.root, split)
            if "imgs_fake" in split:
                self.annotations.update(self._read_cvat_annotations(
                    os.path.join(self.root, f'ann/annotations_{split.replace("imgs_fake/", "")}.xml')))
            elif "backgrounds" == split:
                pass
            else:
                self.annotations.update(self._read_cvat_annotations(os.path.join(self.root, 'ann/annotations.xml')))

            file_ids = glob(image_dir + '/*.jpg')
            file_ids.sort()
            for id in file_ids:
                if self.num_subimg_splits > 0:
                    for i in range(self.num_subimg_splits):
                        for j in range(self.num_subimg_splits):
                            self.files.append(
                                {"file_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
                else:
                    self.files.append({"file_id": id, "sub_img_id": f"{id}_{0}_{0}", "split_x": 0, "split_y": 0})

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
        w_img, h_img = img.shape[0:2]
        w_subimg = math.floor(w_img / self.num_subimg_splits)
        h_subimg = math.floor(h_img / self.num_subimg_splits)
        return img[split_x * w_subimg:split_x * w_subimg + w_subimg, split_y * h_subimg:split_y * h_subimg + h_subimg,
               :]

    def _write_masked_imgs(self, image, label, index):
        image = np.asarray(image, dtype=np.int32)
        label = np.where(label == 1, 120, label)
        image = cv2.addWeighted(label, 1, image, 0.8, 0)
        cv2.imwrite(f"test_output/{index}_masked.jpeg", image)

    def _load_data(self, index):
        # Read image
        subimg_id = self.files[index]
        image_path = subimg_id["file_id"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.float32)

        # Generate mask
        mask_img = np.zeros(image.shape, dtype=np.int32)
        if self.annotations:
            for pol in self.annotations[os.path.basename(image_path)]:
                cv2.fillPoly(mask_img, pts=[pol], color=(1, 1, 1))

        if self.num_subimg_splits > 0:
            image = self._get_sub_img(image, subimg_id["split_x"], subimg_id["split_y"])
            label = (self._get_sub_img(mask_img, subimg_id["split_x"], subimg_id["split_y"]))
        else:
            label = mask_img

        # Make: shortside x longside
        if image.shape[0] > image.shape[1]:
            image = np.swapaxes(image, 0, 1)
            label = np.swapaxes(label, 0, 1)

        # DEBUGGING: Write images for verifying correctness.
        # self._write_masked_imgs(image, label, index)

        # For training, only one channel needed.
        label = label[:, :, 0]
        return image, label, subimg_id["sub_img_id"]


class Rumex(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
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
