import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional

from torch.utils.data import Dataset
from src.data.components.dlib import DLIB
import albumentations as A
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig, OmegaConf
import hydra
import torch, torchvision
from hydra.utils import instantiate
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

class TransformDLIB(Dataset):
    def __init__(self, data: DLIB, transform: Optional[A.Compose] = None):
        self.data = data
        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([
                Normalize(),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    def __getitem__(self, index):
        # TODO: transform image and keypoint using self.transform
        # TODO: convert keypoints range to [0, 1] by diving it with the width and height
        # image shape: (H, W, C)
        sample = self.data[index]
        image = sample['image']
        landmark = sample['landmark']
        # print(landmark)

        image = np.array(image)

        transform = self.transform(image=image, keypoints=landmark)
        image = transform['image'] # (C, H, W)
        landmark = transform['keypoints']
        # print(type(landmark), landmark)

        landmark = landmark / np.array([image.shape[2], image.shape[1]]) # W, H
        # landmark = np.array(landmark)
        # for i in range(landmark.shape[0]):
        #     landmark[i][0] = landmark[i][0] / image.shape[2]
        #     landmark[i][1] = landmark[i][1] / image.shape[1]
        # print(landmark)
        # image = image.permute(1, 2, 0).numpy()
        return image, landmark.astype(np.float32)
    
    @staticmethod
    def tensors_annotation(images: torch.Tensor, keypoints: np.ndarray)->Image:
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD)->torch.Tensor:
            # x: (B, C, H, W)
            tensor = x.clone().permute(1, 2, 3, 0) # (C, H, W, B)
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2) # (B, C, H, W)
        
        images = denormalize(images)
        saved_imgs = []
        for img, keypoint in zip(images, keypoints):
            img = img.permute(1, 2, 0).numpy() * 255 # (H, W, C)
            keypoint = keypoint * np.array([img.shape[1], img.shape[0]]) #W, H
            img = DLIB.image_annotation(Image.fromarray(img.astype(np.uint8)), keypoint)
            saved_imgs.append(ToTensor()(img))
        return torch.stack(saved_imgs)
    
    @staticmethod
    def tensor_annotation(image: torch.Tensor, keypoint: np.ndarray)->Image:
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD)->torch.Tensor:
            tensor = x.clone()
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)
        
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy() * 255
        keypoint = keypoint * np.array([image.shape[1], image.shape[0]]) #W, H
        # print(keypoint)
        DLIB.visual_keypoints(Image.fromarray(image.astype(np.uint8), 'RGB'), keypoint)
        image = DLIB.image_annotation(Image.fromarray(image.astype(np.uint8), 'RGB'), keypoint)
        return image

    def __len__(self):
        return len(self.data)
    
@hydra.main(version_base=None, config_path="../../../configs/", config_name="train.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    transform = instantiate(cfg.data.transform_train)
    print(type(transform))
    transform_dlib = TransformDLIB(DLIB(), transform=transform)
    image, landmark = transform_dlib[20]
    img = DLIB.image_annotation(Image.fromarray(image.numpy(), 'RGB'), landmark)
    img = transform_dlib.tensor_annotation(image, landmark)
    # img.save('test.jpg')
    print(image.shape, landmark.shape)
    # DLIB.visual_keypoints(image, landmark)
if __name__ == "__main__":
    main()