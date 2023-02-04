import cv2

from .models.aggregated_image_detections import AggregatedImageDetections
from .models.image_detection_result import ImageDetectionResult


class BboxPainter(object):
    MULTIPLE_DETECTION_COLOR_BGR = (255, 255, 0)

    def __init__(self, image_path: str) -> None:
        self.cv2_image_bgr = cv2.imread(image_path)

    def draw_boxes(self, aggregated_image_results:AggregatedImageDetections) -> ImageDetectionResult:
        for _, region_detections in aggregated_image_results.image_detection_entries:
            if len(region_detections.detections) > 1:
                bgr_color = BboxPainter.MULTIPLE_DETECTION_COLOR_BGR
            else:
                bgr_color = self._hext_to_bgr(
                    region_detections.detections[0].hex_color)

            border_width = 4
            start, end = region_detections.get_drawable_bbox(border_width)
            cv2.rectangle(self.cv2_image_bgr, start, end, bgr_color, border_width)

        tower_detection = aggregated_image_results.tower_detection
        if tower_detection is not None:
            bgr_color = self._hext_to_bgr(tower_detection.hex_color)
            xmin, ymin, xmax, ymax = [int(x) for x in tower_detection.bbox]
            cv2.rectangle(self.cv2_image_bgr, (xmin, ymin), (xmax, ymax), bgr_color, 4)

        # cv2_image_rgb = cv2.cvtColor(self.cv2_image_bgr, cv2.COLOR_BGR2RGB)
        result = ImageDetectionResult(self.cv2_image_bgr, aggregated_image_results)
        return result

    @staticmethod
    def _hext_to_bgr(hex_color: str):
        rgb_color = hex_color.lstrip('#')
        r, g, b = tuple(int(rgb_color[i:i+2], 16) for i in (0, 2, 4))

        return (b, g, r)

