class ModelLabel(object):
    def __init__(self, name: str, color: str, report_name: str):
        self.detection_label: str = name
        self.hex_color: str = color
        self.report_label_name: str = report_name

    def __repr__(self) -> str:
        return f"Name:{self.detection_label} Color:{self.hex_color} Show In Report:{self.report_label_name}"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def parse(object: dict):
        return ModelLabel(
            object['name'],
            object['color'],
            object['display']
        )
