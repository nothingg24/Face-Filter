import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import notebooks.faceBlendCommon as fbc

VISUALIZE_LANDMARKS = False
SELECTED_KEYPOINTS_INDICES = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]
 
    height, width = img.shape[:-1]
 
    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
 
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0
 
        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))
 
            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y
 
            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')
 
            relevant_keypnts = []
 
            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

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

detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect(img_path: str)->None:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.process(img).detections
    landmark_points = []

    if faces is not None:
        for face in faces:
            face_landmarks_points = []
            bbox = face.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face_cropped = img[y:y+h, x:x+w]

            landmark_points_cropped = face_mesh.process(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
            multi_face_landmarks = landmark_points_cropped.multi_face_landmarks

            if multi_face_landmarks is not None:
                for face_landmarks in multi_face_landmarks:
                    for point in face_landmarks.landmark:
                        x_origin = int(point.x * w) + x
                        y_origin = int(point.y * h) + y

                        face_landmarks_points.append((x_origin, y_origin))
            
            landmark_points.append(face_landmarks_points)
            landmarks = landmark_points[0]

            if VISUALIZE_LANDMARKS:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for point in landmarks:
                    cv2.circle(img, point, 2, (0, 0, 255), -1)
            else:
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # points2 = landmarks
                points2 = []
                for i in SELECTED_KEYPOINTS_INDICES:
                    points2.append(landmarks[i])

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
                img = rgb_img

    return img

if __name__ == '__main__':
    img = detect('IMG_0494.jpg')
    if VISUALIZE_LANDMARKS:
        cv2.imshow('Landmark', img)
        cv2.imwrite('result.png', img)
    else:
        cv2.imshow('Filter', img)
        cv2.imwrite('filter_result.png', img)