import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
import numpy as np
import cv2
from notebooks.tracker_mp import Tracker
from notebooks.kalman import KalmanFilter
import notebooks.faceBlendCommon as fbc
import mediapipe as mp

def track(option: Optional[str] = None):
    capture = cv2.VideoCapture(0)
    if option != '0':
        fourcc = -1 
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    isFirstFrame = True
    detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # tracker1 = Tracker()
    # tracker2 = Tracker()
    tracker = Tracker()

    while capture.isOpened():
        ret, frame = capture.read()
        if option == '0':
            frame = cv2.flip(frame, 1)

        if ret:
            img_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            fps = str(fps)
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            faces = detector.process(img_frame).detections
            # detector.close()

            landmark_points = []

            if faces is not None:
                for face in faces:
                    face_landmarks_points = []
                    bbox = face.location_data.relative_bounding_box
                    ih, iw, _ = img_frame.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    bbox_points = [top_left, bottom_right]
                    # bbox_points = tracker1.track(img_frame, np.array(bbox_points, dtype=np.float32))
                    # cv2.rectangle(frame, (int(bbox_points[0][0]), int(bbox_points[0][1])), (int(bbox_points[1][0]), int(bbox_points[1][1])), (0, 255, 0), 2)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # face_cropped = img_frame[y:y+h, x:x+w]
                    # face_cropped = img_frame[int(bbox_points[0][1]):int(bbox_points[1][1]), int(bbox_points[0][0]):int(bbox_points[1][0])]

                    trackBbox = []
                    if len(trackBbox) == 0:
                        trackBbox = [KalmanFilter(point) for point in bbox_points]
                    else:
                        KalmanFilter.trackpoints(img2GrayPrev, img2Gray, bbox_points, trackBbox)
                    
                    cv2.rectangle(frame, tuple(map(int, trackBbox[0].getPoint())), tuple(map(int, trackBbox[1].getPoint())), (0, 255, 0), 2)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # face_cropped = img_frame[y:y+h, x:x+w]
                    face_cropped = img_frame[int(trackBbox[0].getPoint()[1]):int(trackBbox[1].getPoint()[1]), int(trackBbox[0].getPoint()[0]):int(trackBbox[1].getPoint()[0])]

                    landmark_points_cropped = face_mesh.process(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
                    multi_face_landmarks = landmark_points_cropped.multi_face_landmarks

                    if multi_face_landmarks is not None:
                        for face_landmarks in multi_face_landmarks:
                            for point in face_landmarks.landmark:
                                x_origin = int(point.x * w) + x
                                y_origin = int(point.y * h) + y

                                face_landmarks_points.append((x_origin, y_origin))

                    landmark_points.append(face_landmarks_points)

                    points2 = landmark_points[0]

                    img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if isFirstFrame:
                        img2GrayPrev = np.copy(img2Gray)
                        isFirstFrame = False

                    points2 = tracker.track(img_frame, np.array(points2, dtype=np.float32))
                    for point in points2:
                        cv2.circle(frame, tuple(map(int, point)), radius=1, color=(0, 255, 0), thickness=2)

                    img2GrayPrev = img2Gray
                    cv2.imshow('landmark', frame)

            # face_mesh.close()

            if option != '0':
                out.write(frame)
            else:
                cv2.imshow('frame', frame)

        else:
            break

        keypressed =  cv2.waitKey(1) & 0xFF
        if keypressed == ord('q'):
            break

    capture.release()
    if option != '0':
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track(option=0)