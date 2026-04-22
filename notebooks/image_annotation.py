import pyrootutils
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
from omegaconf import DictConfig
import hydra
from src.models.dlib_module import DLIBLitModule
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from PIL import Image
import os, gdown, onnxruntime
from notebooks.detect import Detection
from src.filter.filter_engine import FilterEngine
from src.filter.filter_config import FILTERS_CONFIG

def download_model(option: int) -> str:
    path = f'checkpoints/{option}'
    if not os.path.exists(path):
        os.makedirs(path)
    ckpt_path = f'{path}/model.ckpt'
    if os.path.isfile(ckpt_path): return ckpt_path
    urls = {
        1: 'https://drive.google.com/file/d/1Xl2xO6mDqA3M9O1L8A_Yy7G-U_fU9n7e/view?usp=sharing',
        2: 'https://drive.google.com/file/d/1qVv6G9F2t5K1k0X9F4p5n7jK2l6m7n8o/view?usp=sharing'
    }
    gdown.download(url=urls[option], output=ckpt_path, quiet=False, use_cookies=False, fuzzy=True)
    return ckpt_path

def get_onnx_model(option: int, cfg: DictConfig) -> str:
    onnx_path = f'checkpoints/{option}/model.onnx'
    if os.path.isfile(onnx_path): return onnx_path
    checkpoint_path = download_model(option)
    net = hydra.utils.instantiate(cfg.net)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net)
    model.to_onnx(file_path=onnx_path, input_sample=torch.rand(1, 3, 224, 224), export_params=True)
    return onnx_path

def detect(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_path = str(cfg.get("image", 'WIN_20240424_16_49_50_Pro.jpg'))
    filter_name = cfg.get("filter", "naruto")
    mode = cfg.get("mode", "onnx")
    option_val = cfg.get("option_val", 2)
    visualize = cfg.get("visualize", False)

    if mode == 'onnx':
        file_path = get_onnx_model(option_val, cfg)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(file_path, sess_options=sess_options, providers=providers)
    else:
        net = hydra.utils.instantiate(cfg.net)
        checkpoint_path = download_model(option_val)
        model = DLIBLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net).to(device)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    detector = Detection()
    faces = detector.detect_faces(image_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    engine = FilterEngine(filter_name)

    print(f"Detected {len(faces)} faces")

    for face in faces:
        bbox = face['facial_area']
        face_img = img_rgb[bbox['y']:bbox['y']+bbox['h'], bbox['x']:bbox['x']+bbox['w']]
        if face_img.size == 0: continue
        transformed = transform(image=face_img)
        transformed_input = transformed['image'].unsqueeze(0)

        if mode == 'onnx':
            ort_inputs = {ort_session.get_inputs()[0].name: transformed_input.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            output = ort_outs[0].squeeze()
        else:
            with torch.inference_mode():
                output = model(transformed_input.to(device)).cpu().squeeze().numpy()
        
        output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])
        landmarks = output.tolist()

        if visualize:
            for p in landmarks:
                cv2.circle(img_bgr, (int(p[0]), int(p[1])), 2, (0, 255, 0), cv2.FILLED)
        else:
            img_bgr = engine.render(img_bgr, landmarks)

    cv2.imshow('Image Annotation', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@hydra.main(version_base=None, config_path="../configs/model", config_name="dlib.yaml")
def main(cfg: DictConfig):
    detect(cfg)

if __name__ == "__main__":
    main()