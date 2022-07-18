from typing import List
import cv2
import torch
import logging
from PIL import Image
from torchvision import transforms

tower_model = ""
tower_label = ["__background__", "MW", "RF", "tower"]
device = ""
score_threshold = 0.8


def _validate_inputs(model_path, image_paths):
    assert isinstance(model_path, str)
    assert isinstance(image_paths, list)
    assert all(isinstance(path, str) for path in image_paths)


def _validate_outputs(results):
    if len(results) == 0:
        return
    assert all(isinstance(result, list) for result in results)
    for bbox in results:
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(coord, int) for coord in bbox)


def _init_model(model_path):
    global tower_model, device

    if tower_model:
        return

    # Select Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        logging.warn(f"Running inference on: {torch.cuda.get_device_name()}")
    # Load Pytorch Model
    tower_model = torch.load(model_path, map_location=device)
    tower_model.eval()


def prepare_model_input(cv_image):
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(img)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(PIL_image)
    img = img.float()
    img = img.unsqueeze(0)

    return img


def get_tower_region(cv_image):
    buffer_image = cv_image.copy()
    image_height, image_width, *_ = buffer_image.shape
    outputs = tower_model(prepare_model_input(buffer_image).to(device))
    
    if len(outputs) == 0:
        pass
    model_output = outputs[0]
    pred_label = model_output['labels'][model_output['scores'] > score_threshold].tolist()

    tower_area = [0,0,0,0]

    for index in range(len(pred_label)):
        label_id = pred_label[index]
        if tower_label[label_id] == "tower":
            pred_box = model_output['boxes'][model_output['scores']
                                             > score_threshold].tolist()
            tower_bbox = pred_box[index]
            xmin, ymin, xmax, ymax = tower_bbox[0], tower_bbox[1], tower_bbox[2], tower_bbox[3]

            # adjust area by adding margin into bbox
            margin = 324
            bbox_xmin = xmin-margin
            bbox_xmax = xmax+margin
            bbox_ymin = ymin-margin
            bbox_ymax = ymax+margin

            # Adjust tower area that are outside image bounds
            if xmin < 0:
                bbox_xmin = 0
            if ymin < 0:
                bbox_ymin = 0
            if xmax > image_width:
                bbox_xmax = image_width
            if ymax > image_height:
                bbox_ymax = image_height

            tower_area = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
            break

    return list(map(int, tower_area))


def process_batch(model_path: str, image_paths: List[str]) -> List[List[List[int]]]:
    _validate_inputs(model_path, image_paths)
    results = []

    _init_model(model_path)

    for image_path in image_paths:
        cv2_image = cv2.imread(image_path)

        tower_area = get_tower_region(cv2_image)
        results.append(tower_area)
    
    _validate_outputs(results)
    return results
