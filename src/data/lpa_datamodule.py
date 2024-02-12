from typing import Any, Dict, Optional, Tuple
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.dlib_lpa import DLIB_LPA
from src.data.components.transform_lpa import TransformDLIB_LPA
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import albumentations as A
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision



class LPADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/300W",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[A.Compose] = None,
        transform_val: Optional[A.Compose] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        pass

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        dataset = DLIB_LPA()
        dataset.prepare_data()
        dataset.prepare_labels()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DLIB_LPA()
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = TransformDLIB_LPA(self.data_train, self.hparams.transform_train)
            self.data_val = TransformDLIB_LPA(self.data_val, self.hparams.transform_val)
            self.data_test = TransformDLIB_LPA(self.data_test, self.hparams.transform_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    @staticmethod
    def batch_visualize(images: torch.Tensor, keypoints: np.ndarray)->None:
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD)->torch.Tensor:
            # x: (B, C, H, W)
            tensor = x.clone().permute(1, 2, 3, 0) # (C, H, W, B)
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2) # (B, C, H, W)
        
        images = denormalize(images)
        fig = plt.figure(figsize=(20, 20))
        for i in range(len(images)):
            image = images[i].permute(1, 2, 0).numpy() * 255
            keypoint = (keypoints[i] + 0.5) * np.array([image.shape[1], image.shape[0]]) #W, H
            ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
            x_values = [x for x, y in keypoint]
            y_values = [y for x, y in keypoint]
            plt.scatter(x_values, y_values, s=10, marker='.', c='r')
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
        plt.savefig('batch.png')    
        plt.show()

@hydra.main(version_base=None, config_path="../../configs/", config_name="train.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    loader = instantiate(cfg.data)
    # print(type(loader))
    loader.setup()
    batch = next(iter(loader.train_dataloader()))
    images, keypoints = batch
    # loader.batch_visualize(images, keypoints)
    print(images.shape, keypoints.shape)
    annotated_batch = TransformDLIB_LPA.tensors_annotation(images, keypoints)
    torchvision.utils.save_image(annotated_batch, 'batch.png')
if __name__ == "__main__":
    main()
