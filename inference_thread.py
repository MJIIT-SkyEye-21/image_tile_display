import numpy as np
from PyQt5 import QtCore
from detection_facade import BoundingBoxGroup, DetectionFacade


class InferenceThread(QtCore.QThread):
    status_update = QtCore.pyqtSignal(str)
    on_result_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, tower_model_path, defect_model_path, image_path):
        super(InferenceThread, self).__init__()
        self.defect_model_path = str(defect_model_path)
        self.tower_model_path = str(tower_model_path)
        self.detector = DetectionFacade(str(image_path))

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
        gp1 = BoundingBoxGroup(
            'defect',
            green_bgr,
            defect_bboxes
        )

        gp_tower = BoundingBoxGroup(
            'tower',
            (255, 255, 0),
            [tower_bbox]
        )

        cv2_image = self.detector.draw_detection_boxes(
            gp_tower,
            [gp1]
        )

        # import cv2
        # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('test.jpg', cv2_image)
        self.on_result_image.emit(cv2_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tower_model_path")
    parser.add_argument("defect_model_path")
    parser.add_argument("image_path")

    th = InferenceThread(**vars(parser.parse_args()))
    th.run()
