import cv2
from typing import Dict, List
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ORANGE_COLOR_BGR = (23,113,237)

class BoundingBoxGroup(object):
    def __init__(self, label, bgr_color, bboxes) -> None:
        self.label = label
        self.bgr_color = bgr_color
        self.bboxes = bboxes


def _is_inside_bbox(det, background):
    xmin, ymin, xmax, ymax = det
    xmin_bg, ymin_bg, xmax_bg, ymax_bg = background
    return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg


def draw_boxes(cv_image, tower_bbox_group: BoundingBoxGroup, detection_box_groups: List[BoundingBoxGroup]):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if len(tower_bbox_group.bboxes) == 0:
        tower_bbox = None
    else:
        tower_bbox = tower_bbox_group.bboxes[0]

    region_bbox_colors = {}
    for box_group in detection_box_groups:
        for box in box_group.bboxes:
            if tower_bbox is not None and not _is_inside_bbox(box, tower_bbox):
                continue

            box = tuple([int(x) for x in box])
            if box not in region_bbox_colors:
                region_bbox_colors[box] = []

            region_bbox_colors[box].append(box_group.bgr_color)

    for box, colors in region_bbox_colors.items():
        # TODO: add label?
        xmin, ymin, xmax, ymax = box
        bgr_color = colors[0]
        
        if len(colors) > 1:
            bgr_color = ORANGE_COLOR_BGR
        
        cv2.rectangle(cv_image, (xmin, ymin),(xmax, ymax), bgr_color, 4)

    if tower_bbox is not None:
        xmin, ymin, xmax, ymax = [int(x) for x in tower_bbox]
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax),
                      tower_bbox_group.bgr_color, 4)

    return cv_image


def find_outside_tower_bboxes(tower_bbox_group: BoundingBoxGroup, defect_bbox_groups: List[BoundingBoxGroup]):
    if len(tower_bbox_group.bboxes) == 0:
        return []
    # TODO: Include skipped bbox type
    outside_bboxes = []
    tower_bbox = tower_bbox_group.bboxes[0]
    for box_group in defect_bbox_groups:
        for box in box_group.bboxes:
            if not _is_inside_bbox(box, tower_bbox):
                outside_bboxes.append(box)

    return outside_bboxes


class DetectionFacade(object):
    def detect_tower(self, tower_model_path: str, image_path: str) -> List[int]:
        import tower_worker
        return tower_worker.main(tower_model_path, image_path)

    def tiled_detect(self, defect_model_path: str, image_path: str, on_status_update: callable) -> List[List[int]]:
        import batch.src.image_tile_display.tiled_worker as tiled_worker
        return tiled_worker.main(
            defect_model_path,
            image_path,
            lambda update_str: on_status_update(update_str)
        )

    def batch_detect_tower(self, tower_model_path: str, image_paths: List[str]):
        from . import tower_worker
        return tower_worker.process_batch(tower_model_path, image_paths)

    def batch_tiled_detect(self, defect_model_path: str, image_paths: List[str]) -> List[List[int]]:
        from . import tiled_worker
        return tiled_worker.process_batch(defect_model_path, image_paths)

    def find_skipped_bboxes(self, tower_bbox_group, defect_bbox_groups: List[BoundingBoxGroup]):
        return find_outside_tower_bboxes(tower_bbox_group, defect_bbox_groups)

    def draw_detection_boxes(self, image_path, tower_bbox_group: BoundingBoxGroup, defect_bbox_groups: List[BoundingBoxGroup]):
        cv2_image = cv2.imread(image_path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        cv2_image = draw_boxes(
            cv2_image,
            tower_bbox_group,
            defect_bbox_groups,
        )

        return cv2_image
