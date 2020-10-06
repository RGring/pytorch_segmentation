import argparse
import os
import numpy as np
import json
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict
import cv2

def save_images(image, mask, output_path, image_file, palette):
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file + '.png'))


def multi_scale_predict(model, image, scales, num_classes, device, normalize, flip=False):
    input_size = (image.shape[0], image.shape[1])
    upsample = nn.Upsample(size=image.shape[:2], mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, input_size[0], input_size[1]))

    for scale in scales:
        new_size = (int(input_size[1]*scale), int(input_size[0]*scale))
        scaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        scaled_img = Image.fromarray(np.uint8(scaled_img))
        input = normalize(transforms.ToTensor()(scaled_img)).unsqueeze(0)
        scaled_prediction = upsample(model(input.to(device)).cpu())

        if flip:
            fliped_img = input.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return torch.from_numpy(total_predictions).to(device).unsqueeze(0)


def main(config_file, model_file, image_folder, output_folder, extension="jpg"):
    config = json.load(open(config_file))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    base_size = config["train_loader"]["args"]["base_size"]

    # Get info from loader
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette


    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
            # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    scales = [1.0]

    image_files = sorted(glob(os.path.join(image_folder, f'*.{extension}')))
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            image = np.asarray(Image.open(img_file).convert('RGB'), dtype=np.float32)
            h, w, _ = image.shape
            upsample = nn.Upsample(size=image.shape[:2], mode='bilinear', align_corners=True)

            # Scale the smaller side to crop size
            if h < w:
                h, w = (base_size, int(base_size * w / h))
            else:
                h, w = (int(base_size * h / w), base_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            prediction = multi_scale_predict(model, image, scales, num_classes, device, normalize)
            prediction = upsample(prediction)
            _, prediction = torch.max(prediction.squeeze(0), 0)
            prediction = prediction.cpu().numpy()
            save_images(image, prediction, output_folder, img_file, palette)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.config, args.model, args.images, args.output, args.extension)
