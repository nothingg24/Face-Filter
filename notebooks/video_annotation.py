import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
from omegaconf import DictConfig
import hydra
from src.models.dlib_module import DLIBLitModule
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
import math, time
from notebooks.kalman import KalmanFilter
import notebooks.faceBlendCommon as fbc
import csv, gdown, os

VISUALIZE_LANDMARKS = False
MODEL_OPTION = 2

filters_config = {
    'naruto':
        [{'path': 'filter/image/naruto.png',
          'anno_path': 'filter/annotations/naruto.csv', #naruto.svg
          'morph': True, 'animated': False, 'has_alpha': True
        }],
}

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

def detect(cfg: DictConfig, option: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (device == 'cpu'):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0

    net = hydra.utils.instantiate(cfg.net)
    checkpoint_path = download_model(MODEL_OPTION)
    model = DLIBLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net=net)
    model = model.to(device) 

    transform = A.Compose([A.Resize(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])

    detector_name = "yolov8"

    capture = cv2.VideoCapture(0)
    if option != '0':
        fourcc = -1 
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    isFirstFrame = True
    sigma = 50
    iter_filter_keys = iter(filters_config.keys())
    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
    count = 0

    while (capture.isOpened()): #True
        ret, frame = capture.read()
        if option == '0':
            frame = cv2.flip(frame, 1)

        if ret:
            img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            fps = str(fps)
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            faces = DeepFace.extract_faces(img_path=img_frame, target_size=(224, 224), detector_backend=detector_name, enforce_detection=False)
            if faces is not None:
                for face in faces:
                    bbox = None
                    old_bbox = face['facial_area']
                    if old_bbox['w'] == int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)):
                        break
                    extend_x = old_bbox['w'] * 0.1
                    extend_y = old_bbox['h'] * 0.1
                    new_bbox = {
                        'x': old_bbox['x'] - extend_x,
                        'y': old_bbox['y'] - extend_y,
                        'w': old_bbox['w'] + 2 * extend_x,
                        'h': old_bbox['h'] + 2 * extend_y
                    }
                    bbox = new_bbox
                    bbox_item = bbox.items()
                    bbox_item = list(bbox_item)

                    top_left = KalmanFilter((bbox['x'], bbox['y']))
                    bottom_right = KalmanFilter((bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
                    bbox_points = [top_left, bottom_right]

                    trackBbox = []
                    if len(trackBbox) == 0:
                        trackBbox = [KalmanFilter(point.getPoint()) for point in bbox_points]
                    else:
                        KalmanFilter.trackpoints(img2GrayPrev, img2Gray, bbox_points, trackBbox)

                    if VISUALIZE_LANDMARKS:
                        cv2.rectangle(frame, tuple(map(int, trackBbox[0].getPoint())), tuple(map(int, trackBbox[1].getPoint())), (0, 255, 0), 2)
                        # cv2.rectangle(frame, (int(bbox['x']), int(bbox['y'])), (int(bbox['x'] + bbox['w']), int(bbox['y'] + bbox['h'])), (0, 255, 0), 2)

                    if (isinstance(img_frame, np.ndarray)):
                        img_frame = Image.fromarray(img_frame)
                    input = img_frame.crop((trackBbox[0].getPoint()[0], trackBbox[0].getPoint()[1], trackBbox[1].getPoint()[0], trackBbox[1].getPoint()[1]))
                    # input = img_frame.crop((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
                    # input = img_frame[int(bbox['y']) : int(bbox['y'] + bbox['h']), int(bbox['x']) : int(bbox['x'] + bbox['w'])]
                    input = np.array(input)
                    transformed = transform(image=input)
                    transformed_input = torch.unsqueeze(transformed['image'], dim=0).to(device)
                    model.eval()
                    with torch.inference_mode():
                        output = model(transformed_input).squeeze()
                    output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])
                    
                    points2 = output.tolist()
                    img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if isFirstFrame:
                        img2GrayPrev = np.copy(img2Gray)
                        isFirstFrame = False

                    trackPoints = []
                    if len(trackPoints) == 0:
                        trackPoints = [KalmanFilter(point) for point in points2]
                    else:
                        KalmanFilter.trackpoints(img2GrayPrev, img2Gray, points2, trackPoints)

                    img2GrayPrev = img2Gray

                    if VISUALIZE_LANDMARKS:
                        for tp in trackPoints:
                            cv2.circle(frame, tuple(map(int, tp.getPoint())), 3, (0, 0, 255) if tp.isPredicted() else (0, 255, 0), cv2.FILLED)
                        cv2.imshow('landmark', frame)
                    else:
                        points3 = [tp.getPoint() for tp in trackPoints]
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
                                    hull2.append(points3[hullIndex[i][0]])
                
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
                                dst_points = [points3[int(list(points1.keys())[0])], points3[int(list(points1.keys())[1])]]
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

                        cv2.imshow('Face Filter', frame)

            if option != '0':
                out.write(frame)
            else:
                cv2.imshow('frame', frame)

        else:
            break

        keypressed =  cv2.waitKey(1) & 0xFF
        if keypressed == ord('q'):
            break
        elif keypressed == ord('f'):
            try:
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            except:
                iter_filter_keys = iter(filters_config.keys())
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            
            count += 1

    capture.release()
    if option != '0':
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")

    @hydra.main(version_base=None, config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        detect(cfg=cfg, option=0)

    main()