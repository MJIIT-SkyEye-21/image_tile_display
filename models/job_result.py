from typing import Dict, List
from .detection import Detection
from .bounding_box_group import BoundingBoxGroup


class JobResult(object):
    def __init__(
        self
    ) -> None:
        self.detections = {}
        self.skipped = {}
        self.regions = {}

    def add_skipped_bboxes(
        self,
        side_name: str,
        image_filename: str,
        out_of_bound_detections: List[Detection]
    ):
        if side_name not in self.skipped:
            self.skipped[side_name] = {}
        self.skipped[side_name][image_filename] = [
            detection.bbox for detection in out_of_bound_detections]

    def add_image_region_summary(
        self,
        side_name: str,
        image_filename: str,
        image_region_summary: Dict[str, int]
    ):
        if side_name not in self.skipped:
            self.skipped[side_name] = {}
        self.skipped[side_name][image_filename] = image_region_summary

    def add_bounding_box_group(
        self,
        side_name: str,
        image_filename: str,
        bounding_box_group: BoundingBoxGroup
    ):
        if side_name not in self.detections:
            self.detections[side_name] = {}

        side_detections = self.detections[side_name]
        if image_filename not in self.detections[side_name]:
            side_detections[image_filename] = {}

        side_image_detections = side_detections[image_filename]
        detection_class_name = bounding_box_group.label

        if detection_class_name not in side_image_detections:
            side_image_detections[detection_class_name] = []

        existing_detections = side_image_detections[detection_class_name]
        box_group_detections = self._summarize_bbox_group(
            bounding_box_group, len(existing_detections))
        existing_detections.extend(box_group_detections)

    def _summarize_bbox_group(self, group: BoundingBoxGroup, initial_counter_value: int):
        counter = initial_counter_value + 1
        enriched = []
        for bbox in group.detections:
            hex_color = group.hex_color
            bbox_type = 'defect' if group.label != 'tower' else 'tower'
            enriched_bbox = self._summarize_detection(
                group.label,
                bbox,
                f'{group.label}_{counter}',
                hex_color,
                bbox_type
            )
            counter += 1
            enriched.append(enriched_bbox)

        return enriched

    def _summarize_detection(
            self,
            bounding_group_class_detection_class_name: str,
            detection: Detection,
            bbox_label: str,
            hex_color: str,
            bbox_type: str,
            score=0
    ):

        return {
            "box": detection.summarize_bbox(),
            "label": bbox_label,
            "name": bounding_group_class_detection_class_name,
            "color": hex_color,
            "type": bbox_type,
            "score": score
        }
