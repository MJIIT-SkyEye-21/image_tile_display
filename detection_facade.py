import cv2
from typing import Dict, List


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

    for box_group in detection_box_groups:
        for box in box_group.bboxes:

            if tower_bbox is not None and not _is_inside_bbox(box, tower_bbox):
                continue

            xmin, ymin, xmax, ymax = [int(x) for x in box]
            # TODO: add label?
            cv2.rectangle(cv_image, (xmin, ymin),
                          (xmax, ymax), box_group.bgr_color, 2)

    if tower_bbox is not None:
        xmin, ymin, xmax, ymax = [int(x) for x in tower_bbox]
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax),
                      tower_bbox_group.bgr_color, 2)

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

    def detect_defects(self, defect_model_path: str, image_path: str, on_status_update: callable) -> List[List[int]]:
        import defect_worker
        return defect_worker.main(
            defect_model_path,
            image_path,
            lambda update_str: on_status_update(update_str)
        )

    def batch_detect_tower(self, tower_model_path: str, image_paths: List[str]):
        from . import tower_worker
        return tower_worker.process_batch(tower_model_path, image_paths)

    def batch_detect_defects(self, defect_model_path: str, image_paths: List[str]) -> List[List[int]]:
        from . import defect_worker
        return defect_worker.process_batch(defect_model_path, image_paths)

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
