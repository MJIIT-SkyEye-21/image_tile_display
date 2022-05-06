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
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    # model_input = transform(image).unsqueeze(0)
    # model_input = model_input.to(device)
    _, image_height, image_width = image.shape
    model_tile_indexes = []
    model_input_tiles = []
    score_threshold = .5
    result_images = []
    out_dir_name = 'tile_output'
    shutil.rmtree(out_dir_name, ignore_errors=True)
    os.makedirs(out_dir_name, exist_ok=True)
    cv2_image = cv2.imread(image_path)

    image_tiles, tiler_crops = get_slices(cv2_image, 224, 0)
    boxes = []
    import pickle
    for (i, (image_tile, tiler_crop)) in enumerate(zip(image_tiles, tiler_crops)):
        # print("@@@@", type(image_tile), tiler_crop)
        xmin, ymin, w, h = tiler_crop
        xmax = xmin + w
        ymax = ymin + h
        # if xmin < (image_width*0.4) or xmin > (image_width*0.7):
        #     continue

        # if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
        #     continue

        # model_tile_indexes.append(i)
        # model_input_tiles.append(transform(image_tile).to(device))
        outputs = model(prepare_model_input(image_tile).to(device))

        # print("@@@@", type(image_tile))
        model_output = outputs[0]
        boxes = model_output['boxes'][model_output['scores'] > score_threshold]
        result_image = draw_result_boxes(image_tile, boxes, score_threshold)

        # cv2.rectangle(cv2_image, (xmin, ymin), (xmax, ymax), (0, 255, 0))
        result_images.append(result_image)
        # break

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

    run_ineference(model, img, image_path, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")

    args = parser.parse_args()
    main(args.model_path, args.image_path)
