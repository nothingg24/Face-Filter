import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import Optional, Callable
from deepface import DeepFace
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import PIL
from PIL import Image
import cv2
import wandb
import torch
import numpy as np

class WandbCallback(Callback):
    def __init__(self, img_path: str):
        super().__init__()
        self.img_path = img_path
    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        frame = cv2.imread(self.img_path)
        transform = A.Compose([A.CenterCrop(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])
        faces = DeepFace.extract_faces(img_path=self.img_path, target_size=(224, 224), detector_backend="ssd")
        for face in faces:
            bbox = face['facial_area'] 
            frame = cv2.rectangle(frame, (bbox['x'], bbox['y']), (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), (0, 255, 0))

            image = face['face']
            transformed = transform(image=image)
            transformed_image = torch.unsqueeze(transformed['image'], dim=0)

            keypoints = trainer.model(transformed_image).detach().numpy()[0]

            h, w, _ = image.shape
            keypoints = ((keypoints) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])).astype(np.uint32)

            for point in keypoints:
                frame = cv2.circle(frame, point, 5, (0, 255, 0), thickness=-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        wandb_logger = WandbLogger(project="Face Filter")
        wandb_logger.log_image(key="Predicted Image", images=[Image.fromarray(frame)])