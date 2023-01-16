import sys
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from pytorch_toolbelt.inference.tiles import ImageSlicer
from .models.detection import Detection

# scriptDir = os.path.dirname(os.path.realpath(__file__))


SCORE_THRESHOLD = 0.8

CLASS_LABEL_LIST = [
    "background",
    "red_peeling",
    "white_peeling",
]

TOWER_CLASS_LABEL_LIST = [
    "background",
    "MW",
    "RF",
    "tower",
]


class TileGeneratorThread():
    # status = QtCore.pyqtSignal(str)
    # image_frame = QtCore.pyqtSignal(np.ndarray)
    # progress = QtCore.pyqtSignal(int)
    # result = QtCore.pyqtSignal(list)

    def __init__(
        self,
        defect_model_path: str,
        tower_model_path: str,
    ):
        # super(TileGeneratorThread, self).__init__()
        # self.image_path = str(image_path)
        self.CLASS_LABEL = CLASS_LABEL_LIST
        self.TOWER_LABEL = TOWER_CLASS_LABEL_LIST
        self.defect_model_path: str = defect_model_path
        self.tower_model_path: str = tower_model_path
        # self.image_name = os.path.basename(self.image_path)
        # self.output_directory = os.path.dirname(str(image_path))
        # self.image_result_directory = os.path.join(scriptDir, "crop_image")
        # os.makedirs(self.image_result_directory, exist_ok=True)
        

    def prepare_model(self):
        # Initialize GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device('cuda')

        # Load Pytorch Model
        # self.model = torch.load(config.MODEL_PATH, map_location=self.device)
        # self.model.eval()
        # self.tower_model = torch.load(
        #     config.TOWER_MODEL_PATH, map_location=self.device)
        # self.tower_model.eval()
        # self.transform = transforms.Compose([transforms.ToTensor()])
        # self.score_threshold = config.SCORE_THRESHOLD
        # self.status.emit("Model Loaded !")
        self.defect_model = torch.load(
            self.defect_model_path, map_location=self.device)
        self.defect_model.eval()
        self.tower_model = torch.load(
            self.tower_model_path, map_location=self.device)
        self.tower_model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.score_threshold = SCORE_THRESHOLD

    def prepare_model_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(img)
        img = self.transform(PIL_image)
        img = img.float()
        img = img.unsqueeze(0)

        return img

    def get_slices(self, cv2_image, tile_side, overlap_factor):
        tile_step = tile_side * (1 - overlap_factor)
        slicer = ImageSlicer(cv2_image.shape,
                             tile_size=(tile_side, tile_side),
                             tile_step=(tile_step, tile_step))
        return slicer.split(cv2_image), slicer.bbox_crops

    def draw_boxes(self, cv_image, processing_box=None, tower_area=None, detection_boxes=[]):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        if processing_box:
            cv2.rectangle(cv_image, (int(processing_box[0]), int(processing_box[1])), (int(
                processing_box[2]), int(processing_box[3])), (255, 255, 0), 4)

        if tower_area:
            cv2.rectangle(cv_image, (int(tower_area[0]), int(tower_area[1])), (int(
                tower_area[2]), int(tower_area[3])), (0, 255, 255), 4)

        for box in detection_boxes:
            cv2.rectangle(cv_image, (int(box[0]), int(
                box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)

        return cv_image

    def get_tower_region(self, cv_image):
        buffer_image = cv_image.copy()
        image_height, image_width, *_ = buffer_image.shape
        outputs = self.tower_model(
            self.prepare_model_input(buffer_image).to(self.device))

        model_output = outputs[0]
        pred_label = model_output['labels'][model_output['scores']
                                            > self.score_threshold].tolist()

        tower_area = []

        for index in range(len(pred_label)):
            label_id = pred_label[index]
            if self.TOWER_LABEL[label_id] in ["tower", "mono_tower"]:
                pred_box = model_output['boxes'][model_output['scores']
                                                 > self.score_threshold].tolist()
                tower_bbox = pred_box[index]
                xmin, ymin, xmax, ymax = tower_bbox[0], tower_bbox[1], tower_bbox[2], tower_bbox[3]

                # adjust area by adding margin into bbox
                margin = 324
                bbox_xmin = xmin-margin
                bbox_xmax = xmax+margin
                bbox_ymin = ymin-margin
                bbox_ymax = ymax+margin

                # Adjust tower area that are outside image bounds
                if xmin < 0:
                    bbox_xmin = 0
                if ymin < 0:
                    bbox_ymin = 0
                if xmax > image_width:
                    bbox_xmax = image_width
                if ymax > image_height:
                    bbox_ymax = image_height

                tower_area = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
                break

        return tower_area

    def is_inside_tower(self, object_box, tower_area):
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = object_box
        tower_xmin, tower_ymin, tower_xmax, tower_ymax = tower_area

        if bbox_xmin < tower_xmin:
            return False
        elif bbox_ymin < tower_ymin:
            return False
        elif bbox_xmax > tower_xmax:
            return False
        elif bbox_ymax > tower_ymax:
            return False
        else:
            return True

    def run(self, image_path: str):
        self.prepare_model()
        # img = read_image(self.image_path)
        img = read_image(image_path)
        _, image_height, image_width = img.shape

        final_result = []
        # cv2_image = cv2.imread(self.image_path)
        cv2_image = cv2.imread(image_path)

        tower_area = self.get_tower_region(cv2_image)
        # if not tower_area:
        #     self.status.emit("Fail to detect tower in the image")
        #     self.progress.emit(100)
        #     return
        if not tower_area:
            raise RuntimeError("Tower not found")

        image_tiles, tiler_crops = self.get_slices(cv2_image, 224, 0)
        # total_tile = len(image_tiles)
        tile_number = 1
        detection_tiles = []

        for image_tile, tiler_crop in zip(image_tiles, tiler_crops):
            xmin, ymin, w, h = tiler_crop
            xmax = xmin + w
            ymax = ymin + h

            bbox_xmin = xmin
            bbox_xmax = xmax
            bbox_ymin = ymin
            bbox_ymax = ymax

            # Adjust tiles that are outside image bounds
            if xmin < 0:
                bbox_xmin = 0
            if ymin < 0:
                bbox_ymin = 0
            if xmax > image_width:
                bbox_xmax = image_width
            if ymax > image_height:
                bbox_ymax = image_height

            processing_box = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]

            if not self.is_inside_tower(processing_box, tower_area):
                # self.progress.emit(int((tile_number/total_tile)*100))
                tile_number += 1
                continue

            # frame_copy = cv2_image.copy()
            # frame_copy = self.draw_boxes(
            #     frame_copy, processing_box, tower_area)

            # self.image_frame.emit(frame_copy)
            # tile_saving_path = os.path.join(
            #     self.image_result_directory, f"{self.image_name}-tile{tile_number}.jpg")
            # cv2.imwrite(tile_saving_path, image_tile)

            # outputs = self.model(self.prepare_model_input(
            #     image_tile).to(self.device))
            outputs = self.defect_model(self.prepare_model_input(
                image_tile).to(self.device))

            model_output = outputs[0]
            pred_score = model_output['scores'][model_output['scores']
                                                > self.score_threshold].tolist()

            if not pred_score:
                continue
            # if pred_score:
            detection_tiles.append(processing_box)
            pred_label = model_output['labels'][model_output['scores']
                                                > self.score_threshold].tolist()
            # pred_box = model_output['boxes'][model_output['scores']
            #                                     > self.score_threshold].tolist()
            max_score = max(pred_score)
            max_idx = pred_score.index(max_score)
            max_label = pred_label[max_idx]
            # max_box = pred_box[max_idx]

            # selected_list = [[self.CLASS_LABEL[max_label], max_score,
            #                     processing_box[0], processing_box[1], processing_box[2], processing_box[3]]]
            detection = Detection(
                max_label,
                processing_box
            )
            detection.detection_class_name = self.CLASS_LABEL[max_label]
            # defect_objects = []

            # for index in range(len(pred_label)):
            #     label = pred_label[index]
            #     score = pred_score[index]
            #     box = pred_box[index]
            #     defect_objects.append(
            #         [self.CLASS_LABEL[label], score, box[0], box[1], box[2], box[3]])
            # else:
            #     defect_objects = [[config.NO_DETECTION_LABEL, 0.0, 0, 0, 0, 0]]
            #     selected_list = [[config.NO_DETECTION_LABEL, 0.0, 0, 0, 0, 0]]

            # result = {"image_path": tile_saving_path, "defect_type": selected_list,
            #           "objects": defect_objects, "g_truth": ""}
            # final_result.append(selected_list)
            final_result.append(detection)            
            # self.progress.emit(int((tile_number/total_tile)*100))
            tile_number += 1

        # self.status.emit("Finished")
        # frame_copy = cv2_image.copy()
        # frame_copy = self.draw_boxes(frame_copy, None, None, detection_tiles)
        # self.image_frame.emit(frame_copy)
        # print(f"saving tile to {self.image_result_directory}")
        # self.result.emit(final_result)
        return final_result

    # def create_class_dic(self, label_path):
    #     with open(label_path) as f:
    #         cls_list = ["background"]
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.replace('\n', '')
    #             cls_list.append(line)

    #     return cls_list
def main(image_path: str, defect_model_path: str, tower_model_path: str):

    g = TileGeneratorThread(defect_model_path, tower_model_path)
    return g.run(image_path)


def process_batch():
    pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
