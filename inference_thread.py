import cv2
import numpy as np
from PyQt5 import QtCore


def _is_inside_bbox(det, background):
    xmin, ymin, xmax, ymax = det
    xmin_bg, ymin_bg, xmax_bg, ymax_bg = background
    return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg


class InferenceThread(QtCore.QThread):
    status_update = QtCore.pyqtSignal(str)
    on_result_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, tower_model_path, defect_model_path, image_path):
        super(InferenceThread, self).__init__()
        self.defect_model_path = str(defect_model_path)
        self.tower_model_path = str(tower_model_path)
        self.image_path = str(image_path)

    def draw_boxes(self, cv_image, detection_boxes, tower_bbox=None):
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

    def _emit_status_update(self, update_string):
        self.status_update.emit(update_string)

    def run(self):
        import tower_worker
        self._emit_status_update("Detecting tower...")
        tower_bbox = tower_worker.main(self.tower_model_path, self.image_path)
        self._emit_status_update("Tower detection complete")

        # import defect_worker
        # detection_tiles = defect_worker.main(
        #     self.defect_model_path,
        #     self.image_path,
        #     lambda update_str: self._emit_status_update(update_str)
        # )
        detection_tiles = None
        cv2_image = cv2.imread(self.image_path)
        cv2_image = self.draw_boxes(
            cv2_image, detection_tiles, tower_bbox=tower_bbox)
        self.on_result_image.emit(cv2_image)

        # # bboxes = [(150, 150, 600, 600)]
        # image = self.draw_boxes(self.image, bboxes)
        # self._display_scaled_image(image)
        # self.event_sink.inference_completed.emit()
