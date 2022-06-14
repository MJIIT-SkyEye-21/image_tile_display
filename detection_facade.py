import cv2
from typing import List


class DefectBoxGroup(object):
    def __init__(self, label, bgr_color, bboxes) -> None:
        self.label = label
        self.bgr_color = bgr_color
        self.bboxes = bboxes


def _is_inside_bbox(det, background):
    xmin, ymin, xmax, ymax = det
    xmin_bg, ymin_bg, xmax_bg, ymax_bg = background
    return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg


def draw_boxes(cv_image, tower_bbox, detection_box_groups: List[DefectBoxGroup]):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    for box_group in detection_box_groups:
        for box in box_group.bboxes:

            if not _is_inside_bbox(box, tower_bbox):
                continue

            xmin, ymin, xmax, ymax = [int(x) for x in box]
            # TODO: add label?
            cv2.rectangle(cv_image, (xmin, ymin),
                          (xmax, ymax), box_group.bgr_color, 2)

    if tower_bbox is not None:
        xmin, ymin, xmax, ymax = [int(x) for x in tower_bbox]
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

    return cv_image


class DetectionFacade(object):
    def __init__(self, image_path) -> None:
        self.image_path = image_path

    def detect_tower(self, tower_model_path: str) -> List[int]:
        import tower_worker
        return tower_worker.main(tower_model_path, self.image_path)

    def detect_defects(self, defect_model_path: str, on_status_update: callable) -> List[List[int]]:
        import defect_worker
        return defect_worker.main(
            defect_model_path,
            self.image_path,
            lambda update_str: on_status_update(update_str)
        )

    def draw_detection_boxes(self, tower_bbox, defect_bbox_groups: List[DefectBoxGroup]):
        cv2_image = cv2.imread(self.image_path)
        cv2_image = draw_boxes(
            cv2_image,
            tower_bbox,
            defect_bbox_groups,

        )

        return cv2_image
