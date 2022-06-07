from pytorch_toolbelt.inference.tiles import ImageSlicer
import torchvision.transforms as T
from torchvision.utils import make_grid, draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import torch
import numpy as np


plt.rcParams["savefig.bbox"] = 'tight'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
INPUT_IMAGE_SIZE = (1000, 800)
SCORE_THRESHOLD = .5


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def display_inference_results(result_images):
    grid = make_grid(result_images, 18)
    show(grid)
    plt.show()


def draw_result_boxes(cv2_image, boxes):

    canvas = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    canvas = T.PILToTensor()(canvas)

    return draw_bounding_boxes(
        canvas,
        boxes=boxes,
        colors="green",
        width=4
    )


def image_to_tensor(img):
    img = img.copy()
    img = cv2.resize(img, INPUT_IMAGE_SIZE)
    img = Image.fromarray(img)
    img = T.ToTensor()(img)
    img = img.float()
    img = img.unsqueeze(0)
    return img


def get_widest_rectangle(bboxes):
    xmins = bboxes[:, 0]
    xmaxs = bboxes[:, 2]
    ymins = bboxes[:, 1]
    ymaxs = bboxes[:, 3]

    return torch.from_numpy(np.array([min(xmins), min(ymins), max(xmaxs), max(ymaxs)]))


def run_ineference(model, cv2_image, device, update_func=None):
    tensor = image_to_tensor(cv2_image)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        output = output[0]
        bboxes = output['boxes'][output['scores'] > SCORE_THRESHOLD].to('cpu')
        widest_rectangle = get_widest_rectangle(bboxes)

    return widest_rectangle


def main(model_path, image_path, update_func=None):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load image as tensor
    # img = read_image(image_path)

    model = torch.load(model_path, map_location=device)
    model.eval()

    cv2_image = cv2.imread(image_path)
    tower_bbox = run_ineference(model, cv2_image, device, update_func)

    if __name__ == '__main__':
        result_image = draw_result_boxes(cv2_image, tower_bbox)
        display_inference_results(result_image)
    else:
        return tower_bbox


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")

    args = parser.parse_args()
    main(args.model_path, args.image_path)