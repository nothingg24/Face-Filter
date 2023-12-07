from typing import Any

import torch, os
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError
import torchvision, wandb, pyrootutils
import numpy as np
from PIL import ImageDraw, Image

# find root of this file
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

# set up output path for drawed batch
outputs_path = path / "test_outputs"

if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)

def draw_batch(images, targets, preds) -> torch.Tensor:
    # helper function
    def denormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
        """Reverse COLOR transform"""
        # clone: make a copy
        # permute: [batch, 3, H, W] -> [3, H, W, batch]
        tmp = images.clone().permute(1, 2, 3, 0)

        # denormalize
        for t, m, s in zip(tmp, mean, std):
            t.mul_(s).add_(m)

        # clamp: limit value to [0, 1]
        # permute: [3, H, W, batch] -> [batch, 3, H, W]
        return torch.clamp(tmp, 0, 1).permute(3, 0, 1, 2)
    
    def annotate_image(image, targets, preds):
        """Draw target & pred landmarks on image"""
        # create an ImageDraw object
        draw = ImageDraw.Draw(image)

        # draw target_landmarks on image (green)
        for x, y in targets:
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 255, 0))

        # draw pred_landmarks on image (red)
        for x, y in preds:
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(255, 0, 0))

        return image
    
    # denormalize
    images = denormalize(images)

    # set an empty list
    images_to_save = []
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # images = images.to(device)
    # targets = targets.to(device)
    # preds = preds.to(device)

    # loop through each sample in batch
    for i, t, p in zip(images, targets, preds):
        # get size of x
        # i = i.cpu().permute(1, 2, 0).numpy() * 255
        i = i.permute(1, 2, 0) * 255
        height, width, _ = i.shape

        # denormalize landmarks -> pixel coordinates
        # t = (t.cpu()) * np.array([width, height])
        # p = (p.cpu()) * np.array([width, height])
        device = t.device
        t = (t + 0.5) * torch.tensor([width, height], device=device)
        device = p.device
        p = (p + 0.5) * torch.tensor([width, height], device=device)

        # draw landmarks on cropped image
        # annotated_image = annotate_image(Image.fromarray(i.astype(np.uint8)), t, p)
        i = torchvision.transforms.functional.to_pil_image(i.to(torch.uint8).permute(2, 0, 1))
        annotated_image = annotate_image(i, t, p)

        # save drawed cropped image
        images_to_save.append(torchvision.transforms.ToTensor()(annotated_image))

    return torch.stack(images_to_save)

class DLIBLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_err = MeanAbsoluteError()
        self.val_err = MeanAbsoluteError()
        self.test_err = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking least so far validation error
        self.val_err_least = MinMetric()
        
        # to make use of all the outputs from each validation_step()
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_err.reset()
        self.val_err_least.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_err(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/err", self.train_err, on_step=False, on_epoch=True, prog_bar=True)

        # return dict with any tensors to read in callback or in `on_train_epoch_end` below
        # return loss or backpropagation will fail
        # return loss
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, images = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_err(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/err", self.val_err, on_step=False, on_epoch=True, prog_bar=True)
        
        # save image, targets and preds to draw batch in on_validation_epoch_end
        self.validation_step_outputs.append({"image": images, "targets": targets, "preds": preds})


        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        err = self.val_err.compute()  # get current val acc
        self.val_err_least(err)  # update best so far val acc
        # log `val_err_least` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/err_least", self.val_err_least.compute(), prog_bar=True)
        
        # get the result of the first batch of validation epoch
        first_val_batch_result = self.validation_step_outputs[0]
        x = first_val_batch_result["image"]
        y = first_val_batch_result["targets"]
        y_hat = first_val_batch_result["preds"]

        # draw the first batch & save it
        annotated_batch = draw_batch(x, y, y_hat)
        torchvision.utils.save_image(annotated_batch, outputs_path / "val_end.png")
        
        # log the first batch
        wandb.log({"annotated_image": wandb.Image(annotated_batch)})
        
        # free memory & prepare for the next validation epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_err(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/err", self.test_err, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        _, preds, _, _ = self.model_step(batch)
        return preds
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # read config file from configs/model/dlib.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    def test_net(cfg):
        net = hydra.utils.instantiate(cfg.model.net)
        print("*"*20+" net "+"*"*20, "\n", net)
        output = net(torch.randn(16, 3, 224, 224))
        print("output", output.shape)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg.model)
        output = module(torch.randn(16, 3, 224, 224))
        print("module output", output.shape)

    @hydra.main(version_base=None, config_path='../../configs/', config_name="train.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_net(cfg)
        test_module(cfg)

    main()
