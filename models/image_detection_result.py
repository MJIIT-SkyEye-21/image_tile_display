from .aggregated_image_detections import AggregatedImageDetections

class ImageDetectionResult(object):
    def __init__(self, cv2_image, aggregated_detections: AggregatedImageDetections) -> None:
        self.detection_regions: AggregatedImageDetections = aggregated_detections
        self.annotated_cv2_image = cv2_image
