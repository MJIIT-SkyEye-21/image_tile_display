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
        self.detector = detection_facade.DetectionFacade(str(image_path))

    def _emit_status_update(self, update_string):
        self.status_update.emit(update_string)

    def run(self):
        self._emit_status_update("Detecting tower...")
        tower_bbox = self.detector.detect_tower(self.tower_model_path)
        self._emit_status_update("Tower detection complete")

        defect_bboxes = self.detector.detect_defects(
            self.defect_model_path,
            lambda update_str: self._emit_status_update(update_str)
        )

        green_bgr = (0, 255, 0)
        bbox_group = detection_facade.DefectBoxGroup(
            'defect',
            green_bgr,
            defect_bboxes
        )

        cv2_image = self.detector.draw_detection_boxes(
            tower_bbox,
            [bbox_group]
        )

        self.on_result_image.emit(cv2_image)
