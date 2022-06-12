import cv2


def _is_inside_bbox(det, background):
    xmin, ymin, xmax, ymax = det
    xmin_bg, ymin_bg, xmax_bg, ymax_bg = background
    return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg


def draw_boxes(cv_image, detection_boxes, tower_bbox=None):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    if detection_boxes is not None:
        for box in detection_boxes:

            if tower_bbox is not None and not _is_inside_bbox(box, tower_bbox):
                continue

            cv2.rectangle(cv_image, (int(box[0]), int(
                box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    if tower_bbox is not None:
        tower_bbox = tower_bbox.numpy()
        xmin, ymin, xmax, ymax = tower_bbox
        cv2.rectangle(cv_image, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), (255, 255, 0), 2)

    return cv_image


class DetectionFacade(object):
    def __init__(self) -> None:
        self.tower_bbox = None
        self.detection_tiles = None

    def detect_tower(self, tower_model_path, image_path) -> None:
        import tower_worker
        self.tower_bbox = tower_worker.main(tower_model_path, image_path)

    def detect_defects(self, defect_model_path: str, image_path: str, on_status_update: callable) -> None:
        import defect_worker
        self.detection_tiles = defect_worker.main(
            defect_model_path,
            image_path,
            lambda update_str: on_status_update(update_str)
        )

    def draw_detection_boxes(self, image_path):
        cv2_image = cv2.imread(image_path)
        cv2_image = draw_boxes(
            cv2_image,
            self.detection_tiles,
            tower_bbox=self.tower_bbox
        )

        return cv2_image
