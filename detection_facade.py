from typing import List
from .models.model_label import ModelLabel
from .models.bounding_box_group import BoundingBoxGroup
from .models.image_detection_result import ImageDetectionResult
from .bbox_painter import BboxPainter
from .detection_aggregator import DetectionAggregator

class DetectionFacade(object):
    def batch_detect_tower(self, tower_model_path: str, image_paths: List[str]):
        from . import tower_worker
        return tower_worker.process_batch(tower_model_path, image_paths)

    def batch_detect_tiled(
        self,
        tower_model_path: str,
        defect_model_path: str,
        image_paths: List[str],
        labels: List[ModelLabel]
    ):
        from . import tiled_detector as td
        return td.process_batch(
            tower_model_path,
            defect_model_path,
            image_paths,
            labels
        )

    def draw_detection_boxes(self, image_path, tower_bbox_group: BoundingBoxGroup, defect_bbox_groups: List[BoundingBoxGroup]) -> ImageDetectionResult:
        painter = BboxPainter(image_path)
        aggregated_detections = DetectionAggregator().aggregate_tower_detections(tower_bbox_group,defect_bbox_groups)
        return painter.draw_boxes(aggregated_detections)
