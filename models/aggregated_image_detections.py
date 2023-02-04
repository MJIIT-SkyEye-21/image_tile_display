from typing import Union, Dict, List, Tuple
from .image_region_bboxes import ImageRegionBBoxes
from .detection import Detection


class AggregatedImageDetections(object):
    def __init__(
        self,
        tower_detection: Union[Detection, None],
    ) -> None:

        self.tower_detection: Union[Detection, None] = tower_detection
        self.image_region_bboxes: Dict[Tuple[int], ImageRegionBBoxes] = {}

    def add_detection(self, region_tile: Tuple[int], detection: Detection):
        if region_tile not in self.image_region_bboxes:
            self.image_region_bboxes[region_tile] = ImageRegionBBoxes(
                detection.bbox)

        self.image_region_bboxes[region_tile].add_bbox(detection)

    @property
    def image_detection_entries(self):
        return self.image_region_bboxes.items()
