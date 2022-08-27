class Detection(object):
    def __init__(self, label_id, bbox):
        assert isinstance(label_id, int)
        assert isinstance(bbox, list)

        self.label_id = label_id
        self.bbox = bbox
    
    def __repr__(self) -> str:
        return f'LabelId:{self.label_id} BBox:{self.bbox}'
    
    def __str__(self) -> str:
        return self.__repr__()