from ultralytics import YOLO
import os
from typing import List
import gdown
import onnxruntime
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch

class Detection:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = self.download_model()
        self.model = YOLO(self.model_path)
        self.onnx_path = self.build_onnx()
        # self.onnx_model = YOLO(self.onnx_path)

    def download_model(self) -> str:
        if not os.path.exists(f'checkpoints/detect'):
            os.makedirs(f'checkpoints/detect')
        
        if os.path.isfile(f'checkpoints/detect/{self.model_name}.pt'):
            return f'checkpoints/detect/{self.model_name}.pt'
        
        if self.model_name == 'yolov8n-face':
            url = 'https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing'
        
        if self.model_name == 'yolov8-lite-t':
            url = 'https://drive.google.com/file/d/1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94/view?usp=sharing'

        if self.model_name == 'yolov8-lite-s':
            url = 'https://drive.google.com/file/d/1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq/view?usp=sharing'

        gdown.download(url=url,output=f'checkpoints/detect/{self.model_name}.pt',quiet=False,use_cookies=False,fuzzy=True)
        return f'checkpoints/detect/{self.model_name}.pt'
    
    def build_onnx(self) -> str:
        if os.path.isfile(f'checkpoints/detect/{self.model_name}.onnx'):
            return f'checkpoints/detect/{self.model_name}.onnx'
        self.model.export(format='onnx')
        onnx_path = f'checkpoints/detect/{self.model_name}.onnx'
        return onnx_path
    
    def detect_faces(self, img_path: str) -> List[dict]:
        faces = []
        results = self.model.predict(img_path, verbose=False, show=False, conf=0.25)[0]

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            right_eye = result.keypoints.xy[0][0].tolist()
            left_eye = result.keypoints.xy[0][1].tolist()

            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

            face = {
                "facial_area": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                },
                "confidence": round(confidence, 2)
            }

            faces.append(face)
        
        return faces
    
    def detect_faces_onnx(self, img_path: str) -> List[dict]:
        faces = []
        transform = A.Compose([A.Resize(640, 640),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ])
        ort_session = onnxruntime.InferenceSession(self.onnx_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = transform(image=img)['image']
        img = torch.unsqueeze(img, dim=0).numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        results = ort_outs[0].squeeze()
        for result in results:
            x, y, w, h = result[:, :4]
            confidence = result[:, 4]

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

            face = {
                "facial_area": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                },
                "confidence": round(confidence, 2)
            }

            faces.append(face)
        
        return faces
