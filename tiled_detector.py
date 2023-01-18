import sys
from typing import List
import cv2
import torch
import logging
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from pytorch_toolbelt.inference.tiles import ImageSlicer
from .models.detection import Detection
worker_logger = logging.getLogger('tiled_detector')

SCORE_THRESHOLD = 0.8

TOWER_CLASS_LABEL_LIST = [
    "background",
    "MW",
    "RF",
    "tower",
]


class TiledDetector():
    def __init__(
        self,
        defect_model_path: str,
        tower_model_path: str,
    ):
        self.TOWER_LABEL = TOWER_CLASS_LABEL_LIST
        self.defect_model_path: str = defect_model_path
        self.tower_model_path: str = tower_model_path

    def prepare_model(self):
        # Initialize GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device = torch.device('cuda')
        self.defect_model = torch.load(
            self.defect_model_path, map_location=self.device)
        self.defect_model.eval()
        self.tower_model = torch.load(
            self.tower_model_path, map_location=self.device)
        self.tower_model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.score_threshold = SCORE_THRESHOLD

    def prepare_model_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(img)
        img = self.transform(PIL_image)
        img = img.float()
        img = img.unsqueeze(0)

        return img

    def get_slices(self, cv2_image, tile_side, overlap_factor):
        tile_step = tile_side * (1 - overlap_factor)
        slicer = ImageSlicer(cv2_image.shape,
                             tile_size=(tile_side, tile_side),
                             tile_step=(tile_step, tile_step))
        return slicer.split(cv2_image), slicer.bbox_crops

    def get_tower_region(self, cv_image):
        buffer_image = cv_image.copy()
        image_height, image_width, *_ = buffer_image.shape
        outputs = self.tower_model(
            self.prepare_model_input(buffer_image).to(self.device))

        model_output = outputs[0]
        pred_label = model_output['labels'][model_output['scores']
                                            > self.score_threshold].tolist()

        tower_area = []

        for index in range(len(pred_label)):
            label_id = pred_label[index]
            if self.TOWER_LABEL[label_id] in ["tower", "mono_tower"]:
                pred_box = model_output['boxes'][model_output['scores']
                                                 > self.score_threshold].tolist()
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

        return tower_area

    def is_inside_tower(self, object_box, tower_area):
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

    def run(self, image_path: str) -> List[Detection]:
        self.prepare_model()
        img = read_image(image_path)
        _, image_height, image_width = img.shape

        final_result = []
        cv2_image = cv2.imread(image_path)

        tower_area = self.get_tower_region(cv2_image)

        if not tower_area:
            worker_logger.warning(f"No tower detected: {image_path}")
            return []

        image_tiles, tiler_crops = self.get_slices(cv2_image, 224, 0)

        tile_number = 1
        detection_tiles = []

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

            processing_box = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]

            if not self.is_inside_tower(processing_box, tower_area):
                tile_number += 1
                continue

            outputs = self.defect_model(self.prepare_model_input(
                image_tile).to(self.device))

            model_output = outputs[0]
            pred_score = model_output['scores'][model_output['scores']
                                                > self.score_threshold].tolist()

            if not pred_score:
                continue

            detection_tiles.append(processing_box)
            pred_label = model_output['labels'][model_output['scores']
                                                > self.score_threshold].tolist()

            max_score = max(pred_score)
            max_idx = pred_score.index(max_score)
            max_label = pred_label[max_idx]

            if max_label == 0:
                continue

            detection = Detection(
                max_label,
                processing_box
            )

            final_result.append(detection)
            tile_number += 1

        return final_result


def main(image_path: str, tower_model_path: str, defect_model_path: str):
    g = TiledDetector(defect_model_path, tower_model_path)
    return g.run(image_path)


def process_batch(
    tower_model_path: str,
    defect_model_path: str,
    image_paths: List[str]
) -> List[List[Detection]]:
    results = []
    g = TiledDetector(defect_model_path, tower_model_path)
    tiled_detection_count = 0
    for image_path in image_paths:
        g.run(image_path)
        defect_areas = g.run(image_path)
        tiled_detection_count += len(defect_areas)

        # Append results for each image, even if they're empty
        # to denote that this image has no corresponding detections
        results.append(defect_areas)

    worker_logger.info(
        f'Model: {defect_model_path} found {tiled_detection_count} detections in {len(image_paths)} images')

    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
