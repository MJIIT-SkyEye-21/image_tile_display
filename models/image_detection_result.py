from typing import Dict, Tuple
from .image_region_bboxes import ImageRegionBBoxes


class ImageDetectionResult(object):
    def __init__(self, cv2_image, detection_regions: Dict[Tuple[int], ImageRegionBBoxes]) -> None:

        self.detection_regions: Dict[Tuple[int],
                                     ImageRegionBBoxes] = detection_regions
        self.annotated_cv2_image = cv2_image

    def summarize_region_detections(self) -> dict:
        detections = {}
        for region_bboxes in self.detection_regions.values():
            for detection in region_bboxes.detections:
                if not detection.detection_class_name in detections:
                    detections[detection.detection_class_name] = 0
                detections[detection.detection_class_name] += 1

        return detections
