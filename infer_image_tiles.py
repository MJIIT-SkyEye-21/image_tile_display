import sys
from inference_thread import InferenceThread
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel, QGridLayout, QFormLayout,
                             QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFrame, QLineEdit
                             )


class UiEventSink(QtCore.QObject):
    model_loaded = QtCore.pyqtSignal()
    image_loaded = QtCore.pyqtSignal()
    inference_started = QtCore.pyqtSignal()
    inference_completed = QtCore.pyqtSignal()


class MainWindow(QWidget):

    def __init__(self, image_path=None, tower_model_path=None, defect_model_path=None):
        super().__init__()
        self.defect_model_path = defect_model_path
        self.tower_model_path = tower_model_path
        self.image_path = image_path

        self.event_sink: UiEventSink = UiEventSink()
        self.event_sink.model_loaded.connect(self.on_model_loaded)
        self.event_sink.image_loaded.connect(self.on_image_loaded)
        self.event_sink.inference_started.connect(self.on_start_inference)
        # self.event_sink.inference_completed.connect(self.on_inference_complete)

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

        self.load_tower_model_button = QPushButton("Load tower model")
        self.load_tower_model_button.clicked.connect(
            self.select_model_path("tower_model_path")
        )
        self.load_tower_model_button.setDisabled(True)

        self.load_defect_model_button = QPushButton("Load defect model")
        self.load_defect_model_button.clicked.connect(
            self.select_model_path("defect_model_path")
        )
        self.load_defect_model_button.setDisabled(True)

        grid.addWidget(load_image_button)
        grid.addWidget(self.load_tower_model_button)
        grid.addWidget(self.load_defect_model_button)

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

        if not file:
            return

        print('Files:', file)

        cv_image = cv2.imread(file)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        self.image = cv_image
        self.image_path = file
        self.event_sink.image_loaded.emit()
        self._display_scaled_image(self.image)

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

    def select_model_path(self, model_variable_name):
        def load_model_action():
            file = self._open_file_dialog(
                ["Models (*.pth)"]
            )
            if not file:
                return
            self.__dict__[model_variable_name] = file
            self.event_sink.model_loaded.emit()

            print(f'{model_variable_name} model:', file)

        return load_model_action

    def on_image_loaded(self):
        # Update image path label
        self.load_defect_model_button.setEnabled(True)
        self.load_tower_model_button.setEnabled(True)

    def on_model_loaded(self):
        # Update model path label
        self.inference_button.setEnabled(True)

    def on_start_inference(self):
        self.inference_button.setText("Inference in progress...")
        self.inference_button.setDisabled(True)

    @QtCore.pyqtSlot(np.ndarray)
    def on_inference_complete(self, cv_image):
        self.inference_button.setText("Start inference")
        self.inference_button.setDisabled(False)
        self._display_scaled_image(cv_image)

    def _open_file_dialog(self, name_filters):
        dlg = QFileDialog()
        dlg.setNameFilters(name_filters)
        if dlg.exec_():
            return dlg.selectedFiles()[0]

    def start_inference(self):
        # Disable inference button
        self.event_sink.inference_started.emit()
        self.inference_worker = InferenceThread(
            self.tower_model_path,
            self.defect_model_path,
            self.image_path
        )
        self.inference_worker.status_update.connect(self.set_status)
        self.inference_worker.on_result_image.connect(
            self.on_inference_complete)
        self.inference_worker.start()

    @QtCore.pyqtSlot(str)
    def set_status(self, status):
        self.inference_button.setText(status)
        if status == "Finished":
            self.event_sink.inference_completed.emit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tower_model_path")
    parser.add_argument("-d", "--defect_model_path")
    parser.add_argument("-i", "--image_path")

    app = QApplication(sys.argv)
    ex = MainWindow(**vars(parser.parse_args()))
    sys.exit(app.exec_())
