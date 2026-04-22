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
import notebooks.faceBlendCommon as fbc
import os, gdown, onnxruntime
from notebooks.detect import Detection
from notebooks.tracker import Tracker
from src.filter.filter_engine import FilterEngine
from src.filter.filter_config import FILTERS_CONFIG

def download_model(option: int) -> str:
    if not os.path.exists(f'checkpoints/{option}'):
        os.makedirs(f'checkpoints/{option}')
    
    ckpt_path = f'checkpoints/{option}/last.ckpt'
    if os.path.isfile(ckpt_path):
        return ckpt_path
    
    urls = {
        1: 'https://drive.google.com/file/d/1-iQWYx2BW0OkiQlN6W78avS8e1d-2Xhc/view?usp=sharing',
        2: 'https://drive.google.com/file/d/17dyMWOMivfqrmvojgl2fs6ul0v1MEAVS/view?usp=sharing'
    }
    
    gdown.download(url=urls[option], output=ckpt_path, quiet=False, use_cookies=False, fuzzy=True)
    return ckpt_path

def get_onnx_model(option: int, cfg: DictConfig) -> str:
    onnx_path = f'checkpoints/{option}/model.onnx'
    if os.path.isfile(onnx_path):
        return onnx_path
    
    checkpoint_path = download_model(option)
    net = hydra.utils.instantiate(cfg.net)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net)
    
    if not os.path.exists(f'checkpoints/{option}'):
        os.makedirs(f'checkpoints/{option}')
    
    model.to_onnx(file_path=onnx_path, input_sample=torch.rand(1, 3, 224, 224), export_params=True)
    return onnx_path

def smooth_bbox(old_bbox, new_bbox, alpha=0.7):
    """Simple EMA smoothing for bounding boxes to reduce jitter."""
    if old_bbox is None:
        return new_bbox
    smoothed = {
        'x': int(alpha * old_bbox['x'] + (1 - alpha) * new_bbox['x']),
        'y': int(alpha * old_bbox['y'] + (1 - alpha) * new_bbox['y']),
        'w': int(alpha * old_bbox['w'] + (1 - alpha) * new_bbox['w']),
        'h': int(alpha * old_bbox['h'] + (1 - alpha) * new_bbox['h'])
    }
    return smoothed

def detect(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mode = cfg.get("mode", "onnx")
    option_val = cfg.get("option_val", 2)
    visualize = cfg.get("visualize", False)
    source_val = str(cfg.get("source", "0"))
    filter_name = cfg.get("filter", "naruto")
    
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
    tracker = Tracker()
    engine = FilterEngine(filter_name)
    
    source = int(source_val) if source_val.isdigit() else source_val
    capture = cv2.VideoCapture(source)
    
    if source_val != '0':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    DETECT_EVERY_N = 3
    frame_count = 0
    last_faces = None
    smoothed_face_bboxes = {} # Track smoothed bboxes for multiple faces if needed

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
            
        if source_val == '0':
            frame = cv2.flip(frame, 1)

        img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % DETECT_EVERY_N == 0 or last_faces is None:
            last_faces = detector.detect_face_video(img_frame)
        faces = last_faces
        frame_count += 1
        
        if faces is not None and len(faces) > 0:
            for i, face in enumerate(faces):
                old_bbox_raw = face['facial_area']
                if old_bbox_raw['w'] == int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)):
                    break
                    
                # Smooth the raw detection bbox before extending it
                smoothed_raw = smooth_bbox(smoothed_face_bboxes.get(i), old_bbox_raw, alpha=0.8)
                smoothed_face_bboxes[i] = smoothed_raw
                
                extend_x, extend_y = smoothed_raw['w'] * 0.1, smoothed_raw['h'] * 0.1
                bbox = {
                    'x': int(max(0, smoothed_raw['x'] - extend_x)),
                    'y': int(max(0, smoothed_raw['y'] - extend_y)),
                    'w': int(smoothed_raw['w'] + 2 * extend_x),
                    'h': int(smoothed_raw['h'] + 2 * extend_y)
                }
                h_f, w_f = frame.shape[:2]
                bbox['w'] = int(min(w_f - bbox['x'], bbox['w']))
                bbox['h'] = int(min(h_f - bbox['y'], bbox['h']))
                
                face_img = img_frame[bbox['y']:bbox['y']+bbox['h'], bbox['x']:bbox['x']+bbox['w']]
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
                
                # The Tracker now uses an improved OneEuroFilter internally
                points3 = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), output).tolist()

                if visualize:
                    for p in points3:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), cv2.FILLED)
                else:
                    frame = engine.render(frame, points3)
        
        cv2.imshow('Face Filter', frame)
        if source_val != '0':
            out.write(frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            keys = list(FILTERS_CONFIG.keys())
            filter_name = keys[(keys.index(filter_name) + 1) % len(keys)]
            engine = FilterEngine(filter_name)
            print(f"Switched to filter: {filter_name}")
            
    capture.release()
    if source_val != '0': out.release()
    cv2.destroyAllWindows()

@hydra.main(version_base=None, config_path="../configs/model", config_name="dlib.yaml")
def main(cfg: DictConfig):
    detect(cfg=cfg)

if __name__ == "__main__":
    main()