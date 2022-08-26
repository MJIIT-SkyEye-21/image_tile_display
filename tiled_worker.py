from typing import List
from unittest import result
import cv2
import torch
import logging
from PIL import Image
from torchvision import transforms
from pytorch_toolbelt.inference.tiles import ImageSlicer

worker_logger = logging.getLogger('tiled_worker')
label = ["corrosion", "peeling"]
score_threshold = 0.8


def _validate_inputs(model_path, image_paths):
    assert isinstance(model_path, str)
    assert isinstance(image_paths, list)
    assert all(isinstance(path, str) for path in image_paths)


def _validate_outputs(results):
    if len(results) == 0:
        return
    assert all(isinstance(result, list) for result in results)
    for image_results in results:
        for bbox in image_results:
            assert isinstance(bbox, list)
            assert len(bbox) == 4
            assert all(isinstance(coord, int) for coord in bbox)


def _init_model(model_path):
    # Select Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        worker_logger.warn(f"Running inference on: {torch.cuda.get_device_name()}")
    # Load Pytorch Model
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def prepare_model_input(cv_image):
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(PIL_image)
    img = img.float()
    img = img.unsqueeze(0)

    return img


def get_slices(cv2_image, tile_side, overlap_factor):
    tile_step = tile_side * (1 - overlap_factor)
    slicer = ImageSlicer(cv2_image.shape,
                         tile_size=(tile_side, tile_side),
                         tile_step=(tile_step, tile_step))
    return slicer.split(cv2_image), slicer.bbox_crops


def is_inside_tower(object_box, tower_area):
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = object_box
    tower_xmin, tower_ymin, tower_xmax, tower_ymax = tower_area

    if bbox_xmin < tower_xmin:
        return False
    elif bbox_ymin < tower_ymin:
        return False
    elif bbox_xmax > tower_xmax:
        return False
    elif bbox_ymax > tower_ymax:
        return False
    else:
        return True


def get_defect_area(cv_image, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    defect_area = []
    image_height, image_width, *_ = cv_image.shape
    image_tiles, tiler_crops = get_slices(cv_image, 224, 0)

    for image_tile, tiler_crop in zip(image_tiles, tiler_crops):
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

        processing_box = (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        outputs = model(prepare_model_input(image_tile).to(device))

        model_output = outputs[0]
        # TODO: Add label to result
        pred_score = model_output['scores'][model_output['scores']
                                            > score_threshold].tolist()

        if pred_score:
            defect_area.append(list(map(int, processing_box)))

    return defect_area


def process_batch(model_path: str, image_paths: List[str]) -> List[List[List[int]]]:
    _validate_inputs(model_path, image_paths)
    results = []

    model = _init_model(model_path)
    bbox_count = 0
    for image_path in image_paths:
        cv2_image = cv2.imread(image_path)
        defect_area = get_defect_area(cv2_image, model)
        bbox_count += len(defect_area)
        results.append(defect_area)
    
    worker_logger.warn(f'Model: {model_path} found {bbox_count} detections')
    _validate_outputs(results)
    del model
    return results
