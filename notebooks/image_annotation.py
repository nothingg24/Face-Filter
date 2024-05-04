import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
from omegaconf import DictConfig
import hydra
from src.models.dlib_module import DLIBLitModule
# from deepface import DeepFace
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
import xml.etree.ElementTree as ET
import notebooks.faceBlendCommon as fbc
import csv
import gdown
import os
import onnx, onnxruntime
from notebooks.detect import Detection

VISUALIZE_LANDMARKS = True
MODEL_OPTION = 2
INFERENCE_MODE = 'onnx'

filters_config = {
    'naruto':
        [{'path': 'filter/image/naruto.png',
          'anno_path': 'filter/annotations/naruto.csv', #naruto.svg
          'morph': True, 'animated': False, 'has_alpha': True
        }],
}

def face_detection(face: dict) -> Optional[dict]:
    old_bbox = face['facial_area']
    extend_x = old_bbox['w'] * 0.1
    extend_y = old_bbox['h'] * 0.1
    bbox = {
        'x': old_bbox['x'] - extend_x,
        'y': old_bbox['y'] - extend_y,
        'w': old_bbox['w'] + 2 * extend_x,
        'h': old_bbox['h'] + 2 * extend_y
    }
    return bbox

def inference(img: Image, bbox: dict, model: DLIBLitModule, transform: A.Compose, device: torch.device) -> np.array:
    input = img.crop((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
    input = np.array(input)
    transformed = transform(image=input)
    transformed_input = torch.unsqueeze(transformed['image'], dim=0).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(transformed_input).squeeze()
    output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])
    return output

def visualize_bbox(img: Image, bbox: dict) -> Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']], outline=(0, 255, 0))
    return img

def visualize_landmarks(img: Image, landmarks: np.array) -> Image:
    draw = ImageDraw.Draw(img)
    for point in landmarks:
        draw.ellipse(xy=(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(0, 255, 0))
    return img

def get_filter_landmarks(annotation_path: str) -> np.array:
    xml_data = open(annotation_path, 'r').read()
    root = ET.XML(xml_data)
    circles = root.findall("circle")
    landmarks = []
    # landmarks = {}
    for circle in circles:
        x = float(circle.get('cx'))
        y = float(circle.get('cy'))
        landmarks.append((x, y))
        # landmarks[int(circle.get('data-label-name')) - 1] = (x, y)
    return np.array(landmarks)

def load_filter_landmarks(annotation_file: str) -> np.array:
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points

def get_filter_image(img_path, has_alpha):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
 
    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))
 
    return img, alpha

def find_convex_hull(points: np.array):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    # hullIndex = cv2.convexHull(np.array(points, dtype=np.float32), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])]) #int
 
    return hull, hullIndex

def load_filter(filter_name: str = 'naruto'):
    filters = filters_config[filter_name]
    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}
 
        img1, img1_alpha = get_filter_image(filter['path'], filter['has_alpha'])
 
        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_filter_landmarks(filter['anno_path'])
 
        temp_dict['points'] = points
 
        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)
 
            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)
 
            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt
 
            if len(dt) == 0:
                continue
 
        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap
 
        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime

def download_model(option: int)-> str:
    if os.path.isfile(f'checkpoints/{option}/last.ckpt'):
        print('Model already downloaded')
        return f'checkpoints/{option}/last.ckpt'
    if not os.path.exists(f'checkpoints/{option}'):
        os.makedirs(f'checkpoints/{option}')
    if (option == 1):
        url='https://drive.google.com/file/d/1-iQWYx2BW0OkiQlN6W78avS8e1d-2Xhc/view?usp=sharing'
    if (option == 2):
        url='https://drive.google.com/file/d/17dyMWOMivfqrmvojgl2fs6ul0v1MEAVS/view?usp=sharing'
    gdown.download(url=url,output=f'checkpoints/{option}/last.ckpt',quiet=False,use_cookies=False,fuzzy=True)
    return f'checkpoints/{option}/last.ckpt'

def get_onnx_model(option: int, cfg: DictConfig)-> str:
    if os.path.isfile(f'checkpoints/{option}/model.onnx'):
        print('Model already downloaded')
        return f'checkpoints/{option}/model.onnx'
    if not os.path.exists(f'checkpoints/{option}'):
        os.makedirs(f'checkpoints/{option}')
    model_path = download_model(option=option)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path=model_path, net=hydra.utils.instantiate(cfg.net))
    file_path = f'checkpoints/{option}/model.onnx'
    model.to_onnx(file_path=file_path, input_sample=torch.rand(1, 3, 224, 224), export_params=True)
    return f'checkpoints/{option}/model.onnx'

def inference_onnx(img: Image, bbox: dict, transform: A.Compose, file_path: str) -> np.array:
    input = img.crop((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
    input = np.array(input)
    transformed = transform(image=input)
    transformed_input = torch.unsqueeze(transformed['image'], dim=0)
    ort_session = onnxruntime.InferenceSession(file_path)
    ort_inputs = {ort_session.get_inputs()[0].name: transformed_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0].squeeze()
    output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])
    return output

def detect(img_path: str, cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if (device == 'cpu'):
    #     os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0
    # detector_backend = "yolov8"
    # faces = DeepFace.extract_faces(img_path=img_path, target_size=(224, 224), detector_backend=detector_backend)

    # detector = Detection(model_name='yolov8n-face')
    # faces = detector.detect_faces(img_path=img_path)
    faces = Detection().detect_faces(img_path=img_path)

    img = Image.open(img_path).convert('RGB')
    transform = A.Compose([A.Resize(224, 224),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ])

    if INFERENCE_MODE == 'onnx':        
        file_path = get_onnx_model(MODEL_OPTION, cfg)      
    else:
        net = hydra.utils.instantiate(cfg.net)
        checkpoint_path = download_model(MODEL_OPTION)
        model = DLIBLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net)
        model = model.to(device)


    for face in faces:
        bbox = face_detection(face=face)
        if INFERENCE_MODE == 'onnx':
            landmarks = inference_onnx(img=img, bbox=bbox, transform=transform, file_path=file_path)
        else:
            landmarks = inference(img=img, bbox=bbox, model=model, transform=transform, device=device)

        if VISUALIZE_LANDMARKS:
            img = visualize_bbox(img, bbox)
            img = visualize_landmarks(img, landmarks)
        else:
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # get points2 ~ points on face
            landmarks = np.vstack([landmarks, np.array([landmarks[0][0], int(bbox['y'])])])
            landmarks = np.vstack([landmarks, np.array([landmarks[16][0], int(bbox['y'])])])
            points2 = landmarks.tolist()

            filters, multi_filter_runtime = load_filter('naruto')

            for idx, filter in enumerate(filters):
                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter['morph']:
                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']
    
                    # create copy of frame
                    warped_img = np.copy(frame)
    
                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])
    
                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []
    
                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])
    
                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)
    
                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
    
                    mask2 = (255.0, 255.0, 255.0) - mask1
    
                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                else:
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
    
                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
    
                    mask2 = (255.0, 255.0, 255.0) - mask1
    
                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                frame = output = np.uint8(output)

                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(np.uint8(rgb_img))
                img = pil_img

    return img


if __name__ == "__main__":
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    
    @hydra.main(version_base=None, config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        image = detect(img_path='WIN_20240424_16_49_50_Pro.jpg', cfg=cfg)
        if VISUALIZE_LANDMARKS:
            image.save('result.png')
            plt.imshow(image)
            plt.show()
        else:
            image.save('filter_result.png')
            plt.imshow(image)
            plt.show()
        # print(get_filter_landmarks('filter/annotations/naruto.svg'))
        # img = Image.open('filter/image/naruto.png').convert('RGB')
        # w, h = img.size
        # landmarks = get_filter_landmarks('filter/annotations/naruto.svg')
        # landmarks = (landmarks/ np.array([100, 100])) * np.array([w, h])
        # img = DLIB.image_annotation(img, landmarks)
        # img.save('naruto_landmarks.png')
    
    main()