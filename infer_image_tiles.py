import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel, QGridLayout, QFormLayout,
                             QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QLineEdit
                             )
from PyQt5.QtGui import QPixmap

import cv2


class UiEventSink(QtCore.QObject):
    model_loaded = QtCore.pyqtSignal()
    image_loaded = QtCore.pyqtSignal()
    inference_started = QtCore.pyqtSignal()
    inference_completed = QtCore.pyqtSignal()


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.event_sink: UiEventSink = UiEventSink()
        self.event_sink.model_loaded.connect(self.on_model_loaded)
        self.event_sink.image_loaded.connect(self.on_image_loaded)
        self.event_sink.inference_started.connect(self.on_start_inference)
        self.event_sink.inference_completed.connect(self.on_inference_complete)

        self.image_label = QLabel()
        self.image_label.setGeometry(QtCore.QRect(10, 10, 1280, 720))
        self.image_label.setMinimumSize(QtCore.QSize(480, 480))
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.grid = QGridLayout()
        self.grid.addWidget(self.image_label, 1, 1)
        self.setLayout(self.grid)

        self.grid.addLayout(self._make_action_buttons(), 2, 1)

        self.setGeometry(500, 250, (3840//4), (2160//4)+300)
        self.setWindowTitle("PyQT show image")
        self.show()

    def _make_action_buttons(self):
        action_buttons = QVBoxLayout()

        action_buttons.addLayout(self._make_load_buttons_grid())
        self.inference_button = self._make_start_inference_button()
        action_buttons.addWidget(self.inference_button)
        # action_buttons.addLayout(self._make_path_labels_vbox())

        return action_buttons

    def _make_path_labels_vbox(self):
        # TODO: make this
        self.image_path_label = QLabel("Image path:")
        self.image_path_label.setContentsMargins(0, 0, 0, 0)
        self.model_path_label = QLabel("Model path:")
        self.model_path_label.setContentsMargins(0, 0, 0, 0)

        path_vbox = QVBoxLayout()
        path_vbox.addWidget(self.image_path_label)
        path_vbox.addWidget(self.model_path_label)
        path_vbox.addStretch()

        path_form = QFormLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setDisabled(True)
        path_form.addRow("Image Path:", self.image_path_edit)
        return path_form

    def _make_load_buttons_grid(self):
        grid = QHBoxLayout()
        load_image_button = QPushButton("Load image")
        load_image_button.clicked.connect(self.select_image_path)

        self.load_model_button = QPushButton("Load model")
        self.load_model_button.clicked.connect(self.select_model_path)
        self.load_model_button.setDisabled(True)

        grid.addWidget(load_image_button)
        grid.addWidget(self.load_model_button)

        return grid

    def _make_start_inference_button(self):
        button = QPushButton("Start inference")
        button.setDisabled(True)
        button.clicked.connect(self.start_inference)
        return button

    def select_image_path(self):
        file = self._open_file_dialog(
            ["Images (*.png *.jpg *.jpeg *.bmp *.JPG *.JPEG *.BMP)"]
        )

        print('Files:', file)

        cv_image = cv2.imread(file)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        self.image = cv_image
        self.image_path = file
        self.event_sink.image_loaded.emit()
        self._display_scaled_image(self.image)

        # self.image_label.setPixmap(QPixmap(file))
        # self.setFixedSize(self.grid.sizeHint())

    def _display_scaled_image(self, cv_image):
        self.image_label.clear()
        qImage = QtGui.QImage(
            cv_image.data,
            cv_image.shape[1],
            cv_image.shape[0],
            QtGui.QImage.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(qImage)
        smaller_pixmap = pixmap.scaled(
            pixmap.width()//4,
            pixmap.height()//4,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation
        )

        self.image_label.setPixmap(smaller_pixmap)
        pass

    def select_model_path(self):
        file = self._open_file_dialog(
            ["Models (*.pth)"]
        )
        self.model_path = file
        self.event_sink.model_loaded.emit()

        print('Files:', file)

    def on_image_loaded(self):
        # Update image path label
        self.load_model_button.setEnabled(True)

    def on_model_loaded(self):
        # Update model path label
        self.inference_button.setEnabled(True)

    def on_start_inference(self):
        self.inference_button.setText("Inference in progress...")
        self.inference_button.setDisabled(True)

    def on_inference_complete(self):
        self.inference_button.setText("Start inference")
        self.inference_button.setDisabled(False)

    def _open_file_dialog(self, name_filters):
        dlg = QFileDialog()
        dlg.setNameFilters(name_filters)
        if dlg.exec_():
            return dlg.selectedFiles()[0]

    def start_inference(self):
        # Disable inference button
        self.event_sink.inference_started.emit()
        QtCore.QTimer.singleShot(0, self._run_inference)

    def _run_inference(self):
        import model_worker
        # Boxes are in x1, y1, x2, y2 format
        bboxes = model_worker.main(self.model_path, self.image_path)
        # bboxes = [(150, 150, 600, 600)]
        image = self.draw_boxes(self.image, bboxes)
        self._display_scaled_image(image)
        self.event_sink.inference_completed.emit()

    def draw_boxes(self, cv_image, bboxes):
        bbox_image = cv_image.copy()
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 12)

        return bbox_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
