from turtle import update
import cv2
import numpy as np
from PyQt5 import QtCore
from pytorch_toolbelt.inference.tiles import ImageSlicer

class InferenceThread(QtCore.QThread):
    status_update = QtCore.pyqtSignal(str)
    on_result_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, model_path, image_path):
        super(InferenceThread, self).__init__()
        self.score_threshold = .5
        self.model_path = str(model_path)
        self.image_path = str(image_path)

    def draw_boxes(self, cv_image, detection_boxes, processing_box=None):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        for box in detection_boxes:
            cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) , (0,255,0), 2)

        if processing_box:
            cv2.rectangle(cv_image, (int(processing_box[0]), int(processing_box[1])), (int(processing_box[2]), int(processing_box[3])) , (255,255,0), 2)

        return cv_image

    def _emit_status_update(self, update_string):
        self.status_update.emit(update_string)

    def run(self):
        import model_worker
        detection_tiles = model_worker.main(
            self.model_path, 
            self.image_path, 
            lambda update_str: self._emit_status_update(update_str)
        )

        cv2_image = cv2.imread(self.image_path)
        cv2_image = self.draw_boxes(cv2_image, detection_tiles)
        self.on_result_image.emit(cv2_image)

        # # bboxes = [(150, 150, 600, 600)]
        # image = self.draw_boxes(self.image, bboxes)
        # self._display_scaled_image(image)
        # self.event_sink.inference_completed.emit()
