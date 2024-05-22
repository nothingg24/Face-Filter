import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import onnxruntime
from PIL import Image

from notebooks.tracker import Tracker
from notebooks.detect import Detection

MODEL_PATH = "checkpoints/2/model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)
TRANSFORMS = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
detector = Detection()
tracker = Tracker()

def track(option: Optional[str]='0'):
    capture = cv2.VideoCapture(0)
    if option != '0':
        fourcc = -1 
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            continue
        if option == '0':
            frame = cv2.flip(frame, 1)
    
        img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_face_video(img_frame)
        if faces is None or len(faces) == 0:
            continue
              
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

            cv2.rectangle(frame, (int(bbox['x']), int(bbox['y'])), (int(bbox['x']+bbox['w']), int(bbox['y']+bbox['h'])), (0, 255, 0), 2)

            face_img = img_frame[int(bbox['y']):int(bbox['y']+bbox['h']), int(bbox['x']):int(bbox['x']+bbox['w'])]
            transformed = TRANSFORMS(image=face_img)
            transformed_img = transformed['image']
            transformed_img = transformed_img.unsqueeze(0)
            ort_inputs = {ort_session.get_inputs()[0].name: transformed_img.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            output = ort_outs[0].squeeze()
            output = (output + 0.5) * np.array([bbox['w'], bbox['h']]) + np.array([bbox['x'], bbox['y']])

            landmarks = tracker.track(img_frame, output)
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)
            cv2.imshow('landmark', frame)

        if option != '0':
            out.write(frame)
        else:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    if option != '0':
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track(option=0)
