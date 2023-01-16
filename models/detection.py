from typing import List


class Detection(object):
    def __init__(self, label_id: int, bbox: List[int]):
        assert isinstance(label_id, int)
        assert isinstance(bbox, list)

        self.label_id: int = label_id
        self.bbox: List[int] = [int(x) for x in bbox]
        self.hex_color: str = None
        self.detection_class_name: str = None

    def summarize_bbox(self):
        xmin, ymin, xmax, ymax = self.bbox
        return {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }

    def __repr__(self) -> str:
        return f'LabelId:{self.label_id}({self.detection_class_name}) BBox:{self.bbox}'

    def __str__(self) -> str:
        return self.__repr__()
