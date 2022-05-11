from pytorch_toolbelt.inference.tiles import ImageSlicer
import torchvision.transforms as T
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os
import cv2
import torch
import numpy as np


plt.rcParams["savefig.bbox"] = 'tight'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_slices(cv2_image, tile_side, overlap_factor):
    tile_step = tile_side * (1 - overlap_factor)
    slicer = ImageSlicer(cv2_image.shape,
                         tile_size=(tile_side, tile_side),
                         tile_step=(tile_step, tile_step))
    return slicer.split(cv2_image), slicer.bbox_crops


def draw_result_boxes(cv2_image, boxes, score_threshold):

    canvas = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    canvas = T.PILToTensor()(canvas)

    return draw_bounding_boxes(
        canvas,
        boxes=boxes,
        colors="green",
        width=4
    )


def prepare_model_input(img):
    # img = Image.open(image_path)
    # img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(img)
    img = T.ToTensor()(PIL_image)
    img = img.float()
    img = img.unsqueeze(0)
    img = img.cuda()

    return img


def run_ineference(model, image, image_path, device):
    _, image_height, image_width = image.shape

    score_threshold = .5
    result_images = []
    detection_tiles = []
    cv2_image = cv2.imread(image_path)

    image_tiles, tiler_crops = get_slices(cv2_image, 224, 0)
    boxes = []

    work_units = list(zip(image_tiles, tiler_crops))
    batch_size = 5

    for image_tiles, tiler_crops in np.array_split(work_units, batch_size):
        outputs = model(prepare_model_input(image_tiles).to(device))

        model_output = outputs
        boxes = model_output['boxes'][model_output['scores'] > score_threshold]

        for i, image_tile in enumerate(image_tiles):
            result_image = draw_result_boxes(
                image_tile, boxes[i], score_threshold)
            result_images.append(result_image)

        for tiler_crop, tile_boxes in zip(tiler_crops, boxes):
            if len(tile_boxes) == 0:
                continue
            xmin, ymin, w, h = tiler_crop
            xmax = xmin + w
            ymax = ymin + h

            bbox_xmin = xmin
            bbox_xmax = xmax
            bbox_ymin = ymin
            bbox_ymax = ymax

            # Adjust tiles that are outside image bounds
            if xmin < 0:
                bbox_xmin = 0
            if ymin < 0:
                bbox_ymin = 0
            if xmax > image_width:
                bbox_xmax = image_width
            if ymax > image_height:
                bbox_ymax = image_height

            detection_tiles.append(
                (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

    return result_images, detection_tiles


def display_inference_results(result_images):
    grid = make_grid(result_images, 18)
    show(grid)
    plt.show()


def main(model_path, image_path):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load image as PIL.Image
    img = read_image(image_path)

    model = torch.load(model_path, map_location=device)
    model.eval()

    result_images, detection_tiles = run_ineference(
        model, img, image_path, device)

    if __name__ == '__main__':
        display_inference_results(result_images)
    else:
        return detection_tiles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")

    args = parser.parse_args()
    main(args.model_path, args.image_path)
