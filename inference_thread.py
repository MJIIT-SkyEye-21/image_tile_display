import numpy as np
from PyQt5 import QtCore
import detection_facade


class InferenceThread(QtCore.QThread):
    status_update = QtCore.pyqtSignal(str)
    on_result_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, tower_model_path, defect_model_path, image_path):
        super(InferenceThread, self).__init__()
        self.defect_model_path = str(defect_model_path)
        self.tower_model_path = str(tower_model_path)
        self.image_path = str(image_path)
        self.detector = detection_facade.DetectionFacade()

    def _emit_status_update(self, update_string):
        self.status_update.emit(update_string)

    def run(self):
        self._emit_status_update("Detecting tower...")
        self.detector.detect_tower(
            self.tower_model_path, self.image_path)
        self._emit_status_update("Tower detection complete")

        self.detector.detect_defects(
            self.defect_model_path,
            self.image_path,
            lambda update_str: self._emit_status_update(update_str)
        )

        cv2_image = self.detector.draw_detection_boxes(self.image_path)
        self.on_result_image.emit(cv2_image)
