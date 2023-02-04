from typing import List
from .models.bounding_box_group import BoundingBoxGroup
from .models.detection import Detection
from .models.aggregated_image_detections import AggregatedImageDetections


class DetectionAggregator(object):
    def __init__(self) -> None:
        pass

    def aggregate_tower_detections(self, tower_bbox_group: BoundingBoxGroup, detection_box_groups: List[BoundingBoxGroup]) -> AggregatedImageDetections:
        if len(tower_bbox_group.detections) == 0:
            tower_detection = None
        else:
            tower_detection = tower_bbox_group.detections[0]

        result = AggregatedImageDetections(tower_detection)
        for box_group in detection_box_groups:
            for detection in box_group.detections:
                # Skip detections outside tower area
                if tower_detection is not None and not self._is_inside_bbox(detection, tower_detection.bbox):
                    continue

                # BBOX of detection is the image region if it's a tiled model
                # OR a normal image patch if it's a full image model
                image_region_coordinates = tuple(detection.bbox)
                result.add_detection(image_region_coordinates, detection)

        return result
    @staticmethod
    def _is_inside_bbox(
        detetection: Detection,
        background_bounding_box: List[int]
    ):
        xmin, ymin, xmax, ymax = detetection.bbox
        xmin_bg, ymin_bg, xmax_bg, ymax_bg = background_bounding_box
        return xmin >= xmin_bg and xmax <= xmax_bg and ymin >= ymin_bg and ymax <= ymax_bg
