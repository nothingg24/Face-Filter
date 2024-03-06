import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
from omegaconf import DictConfig
import hydra
from src.models.dlib_module import DLIBLitModule
from deepface.detectors import FaceDetector
from deepface import DeepFace
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from src.data.components.dlib import DLIB
from src.data.components.transform_dlib import TransformDLIB
import matplotlib.pyplot as plt
import cv2
import PIL
from PIL import Image, ImageDraw

def detect(img_path: str, cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = hydra.utils.instantiate(cfg.net)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path='logs/train/runs/2024-02-22_14-14-53/checkpoints/last.ckpt', net=net)
    model = model.to(device)

    transform = A.Compose([A.Resize(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])

    detector_name = "ssd"

    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    faces = DeepFace.extract_faces(img_path=img_path, target_size=(224, 224), detector_backend=detector_name)
    face = faces[0]

    # face: list = [cropped image (h x w x c, np.array), bounding-box, confidence]
    # bbox = face['facial_area']

    old_bbox = face['facial_area']
    extend_x = old_bbox['w'] * 0.1
    extend_y = old_bbox['h'] * 0.1
    bbox = {
        'x': old_bbox['x'] - extend_x,
        'y': old_bbox['y'] - extend_y,
        'w': old_bbox['w'] + 2 * extend_x,
        'h': old_bbox['h'] + 2 * extend_y
    }

    draw = ImageDraw.Draw(img)
    draw.rectangle([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']], outline=(0, 255, 0))

    # inference
    input = img.crop((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
    input.save('input.png')
    input = np.array(input)
    transformed = transform(image=input)
    transformed_input = torch.unsqueeze(transformed['image'], dim=0).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(transformed_input).squeeze()

    # test
    TransformDLIB.tensor_annotation(transformed['image'], output).save('output.png')

    h, w, _ = input.shape
    # print(output, bbox)

    output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])
    
    # print(output)

    for point in output:
        draw.ellipse(xy=(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(0, 255, 0))

    return img


if __name__ == "__main__":
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    
    @hydra.main(version_base=None, config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        image = detect(img_path='face_img.jpg', cfg=cfg)
        image.save('result.png')
        plt.imshow(image)
        plt.show()
    
    main()