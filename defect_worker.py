from pytorch_toolbelt.inference.tiles import ImageSlicer
import torchvision.transforms as T
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import torch
import numpy as np


plt.rcParams["savefig.bbox"] = 'tight'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
BATCH_SIZE = 10


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


def images_to_tensors(images):
    # img = Image.open(image_path)
    # img = np.array(img)
    cuda_images = []

    for img in images:
        # TODO: should we remove this conversion and
        # perform inference on BGR images?
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(img)
        img = T.ToTensor()(PIL_image)
        img = img.float()
        img = img.unsqueeze(0)
        cuda_images.append(img)

    cuda_images = np.stack(cuda_images)
    cuda_images = np.squeeze(cuda_images)
    return torch.from_numpy(cuda_images)

# def draw_boxes(self, cv_image, bboxes):
#         bbox_image = cv_image.copy()
#         for box in bboxes:
#             x1, y1, x2, y2 = box
#             cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 12)

#         return bbox_image


def run_ineference(model, image, image_path, update_func=None):
    _, image_height, image_width = image.shape

    score_threshold = .5
    result_images = []
    detection_tiles = []
    cv2_image = cv2.imread(image_path)

    image_tiles, tiler_crops = get_slices(cv2_image, 224, 0)
    boxes = []

    work_units = list(zip(image_tiles, tiler_crops))
    num_splits = len(work_units) // BATCH_SIZE
    tiles_processed = 0

    for batch in np.array_split(work_units, num_splits):
        batch_tiles = []
        crop_tiles = []

        for (tile, crop) in batch:
            batch_tiles.append(tile)
            crop_tiles.append(crop)

        batch_tensors = images_to_tensors(batch_tiles)
        model_output = model(batch_tensors.to(_get_device()))

        if update_func:
            tiles_processed += len(batch_tiles)
            status = 'Processed: {}/{}'.format(tiles_processed,
                                               len(work_units))
            update_func(status)

        print('Processed:', len(batch_tiles), 'tiles')

        batch_boxes = []
        for i, image_tile in enumerate(batch_tiles):
            boxes = model_output[i]['boxes'][model_output[i]
                                             ['scores'] > score_threshold]

            # result_image = draw_result_boxes(
            #     image_tile, boxes, score_threshold)
            # result_images.append(result_image)
            batch_boxes.append(boxes)

        for tiler_crop, tile_boxes in zip(crop_tiles, batch_boxes):
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


def _get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _get_model(model_path):
    model = torch.load(model_path, map_location=_get_device())
    model.eval()

    return model


def process_batch(model_path, image_paths):
    model = _get_model(model_path)
    results = {}

    for image_path in image_paths:
        img = read_image(image_path)
        _, detection_tiles = run_ineference(
            model,
            img,
            image_path
        )

        results[image_path] = detection_tiles

    return results


def main(model_path, image_path, update_func=None):
    model = _get_model(model_path)
    # Load image as PIL.Image
    img = read_image(image_path)

    result_images, detection_tiles = run_ineference(
        model,
        img,
        image_path,
        update_func
    )

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
