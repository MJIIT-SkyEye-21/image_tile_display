import cv2
from .models.image_region_bboxes import ImageRegionBBoxes
from .models.bounding_box_group import BoundingBoxGroup
from .models.detection import Detection
from .models.image_detection_result import ImageDetectionResult
from typing import Dict, List, Tuple

MULTIPLE_DETECTION_COLOR_BGR = (255, 255, 0)


def _is_inside_bbox(
    detetection: Detection,
    background_bounding_box: List[int]
):
    xmin, ymin, xmax, ymax = detetection.bbox
    xmin_bg, ymin_bg, xmax_bg, ymax_bg = background_bounding_box
    return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg


def _hext_to_bgr(hex_color: str):
    rgb_color = hex_color.lstrip('#')
    r, g, b = tuple(int(rgb_color[i:i+2], 16) for i in (0, 2, 4))

    return (b, g, r)


def draw_boxes(cv_image, tower_bbox_group: BoundingBoxGroup, detection_box_groups: List[BoundingBoxGroup]) -> ImageDetectionResult:
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if len(tower_bbox_group.detections) == 0:
        tower_detection = None
    else:
        tower_detection = tower_bbox_group.detections[0]

    image_region_bboxes: Dict[Tuple[int], ImageRegionBBoxes] = {}
    for box_group in detection_box_groups:
        for detection in box_group.detections:
            # Skip detections outside tower area
            if tower_detection is not None and not _is_inside_bbox(detection, tower_detection.bbox):
                continue

            # BBOX of detection is the image region if it's a tiled model
            # OR a normal image patch if it's a full image model
            image_region_coordinates = tuple(detection.bbox)
            if detection not in image_region_bboxes:
                image_region_bboxes[image_region_coordinates] = ImageRegionBBoxes(
                    detection.bbox)

            image_region_bboxes[image_region_coordinates].add_bbox(detection)

    for detection, region_detections in image_region_bboxes.items():
        if len(region_detections.detections) > 1:
            bgr_color = MULTIPLE_DETECTION_COLOR_BGR
        else:
            bgr_color = _hext_to_bgr(region_detections.detections[0].hex_color)

        border_width = 4
        start, end = region_detections.get_drawable_bbox(border_width)
        cv2.rectangle(cv_image, start, end, bgr_color, border_width)

    if tower_detection is not None:
        bgr_color = _hext_to_bgr(tower_bbox_group.hex_color)
        xmin, ymin, xmax, ymax = [int(x) for x in tower_detection.bbox]
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), bgr_color, 4)

    result = ImageDetectionResult(cv_image, image_region_bboxes)
    return result


def find_outside_tower_bboxes(tower_bbox_group: BoundingBoxGroup, defect_bbox_groups: List[BoundingBoxGroup]) -> List[Detection]:
    if len(tower_bbox_group.detections) == 0:
        return []
    # TODO: Include skipped bbox type
    outside_bboxes = []
    tower_bbox = tower_bbox_group.detections[0]
    for box_group in defect_bbox_groups:
        for detection in box_group.detections:
            if _is_inside_bbox(detection, tower_bbox.bbox):
                continue
            outside_bboxes.append(detection)

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

    def draw_detection_boxes(self, image_path, tower_bbox_group: BoundingBoxGroup, defect_bbox_groups: List[BoundingBoxGroup]) -> ImageDetectionResult:
        cv2_image = cv2.imread(image_path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        image_detection_result = draw_boxes(
            cv2_image,
            tower_bbox_group,
            defect_bbox_groups,
        )

        return image_detection_result
