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
import matplotlib.pyplot as plt
import cv2

def detect(img_path: str, cfg: DictConfig) -> None:
    net = hydra.utils.instantiate(cfg.net)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path='logs/train/runs/2024-02-22_14-14-53/checkpoints/last.ckpt', net=net)

    transform = A.Compose([A.CenterCrop(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])

    detector_name = "ssd"

    frame = cv2.imread(img_path)

    # detector = FaceDetector.build_model(detector_name)
    # faces = FaceDetector.detect_faces(detector, detector_name, frame)
    faces = DeepFace.extract_faces(img_path=img_path, target_size=(224, 224), detector_backend=detector_name)
    # faces = DeepFace.detectFace(img_path=img_path, target_size=(224, 224), detector_backend=detector_name)

  
    for face in faces:
        # face: list = [cropped image (h x w x c, np.array), bounding-box, confidence]
        bbox = face['facial_area'] 
        frame = cv2.rectangle(frame, (bbox['x'], bbox['y']), (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), (0, 255, 0))

        image = face['face']
        transformed = transform(image=image)
        transformed_image = torch.unsqueeze(transformed['image'], dim=0).cuda()

        keypoints = model(transformed_image).cpu().detach().numpy()[0]

        h, w, _ = image.shape
        print(keypoints, bbox)
        keypoints = ((keypoints) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])).astype(np.uint32)
        print(keypoints)

        for point in keypoints:
            frame = cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)
        # frame = frame * 255
    return frame


if __name__ == "__main__":
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    
    @hydra.main(version_base=None, config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        frame = detect(img_path='IMG_0494.jpg', cfg=cfg)
        plt.imshow(frame)
        plt.show()
        cv2.imwrite('result.png', frame)
    
    main()