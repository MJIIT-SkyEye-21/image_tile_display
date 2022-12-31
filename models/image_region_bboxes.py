from typing import List
from .detection import Detection


class ImageRegionBBoxes(object):
    def __init__(self, bbox_image_region: List[int]) -> None:
        """
            Every region (patch/tile) of the pixel-image has a number Detections
            this class aggregates detections for a *Single* image patch/tile.
        """
        self.bbox_image_region: List[int] = bbox_image_region
        self.detections: List[Detection] = []

    def add_bbox(self, box: Detection):
        self.detections.append(box)
