from typing import List
from .detection import Detection


class BoundingBoxGroup(object):
    def __init__(self, label: str, hex_color: str, detections: List[Detection]) -> None:
        self.label: str = label
        self.hex_color: str = hex_color
        self.detections: List[Detection] = detections

        for detection in self.detections:
            detection.hex_color = hex_color
            detection.detection_class_name = label

    def __repr__(self) -> str:
        return f"Label: {self.label} | Hex: {self.hex_color} | BBoxes:{self.detections}"

    def __str__(self) -> str:
        return self.__repr__()
