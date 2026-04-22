import cv2
import csv
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import notebooks.faceBlendCommon as fbc
from src.filter.filter_config import FILTERS_CONFIG

@dataclass
class FilterRuntime:
    img: np.ndarray
    img_alpha: Optional[np.ndarray]
    points: Dict[str, Tuple[int, int]]
    hull: Optional[List[Tuple[int, int]]] = None
    hull_index: Optional[np.ndarray] = None
    dt: Optional[List[Tuple[int, int, int]]] = None

class FilterEngine:
    def __init__(self, filter_name: str = 'naruto'):
        if filter_name not in FILTERS_CONFIG:
            raise KeyError(f"Filter '{filter_name}' not found in FILTERS_CONFIG")
        
        self.filter_configs = FILTERS_CONFIG[filter_name]
        self.runtimes = self._load_filters()

    def _load_filters(self) -> List[FilterRuntime]:
        runtimes = []
        for config in self.filter_configs:
            # Load image
            img = cv2.imread(config['path'], cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            img_alpha = None
            if config['has_alpha'] and img.shape[2] == 4:
                b, g, r, img_alpha = cv2.split(img)
                img = cv2.merge((b, g, r))
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            # Load landmarks
            points = {}
            with open(config['anno_path'], 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        x, y = int(row[1]), int(row[2])
                        points[row[0]] = (x, y)
                    except (ValueError, IndexError):
                        continue

            runtime = FilterRuntime(img=img, img_alpha=img_alpha, points=points)

            if config['morph']:
                # Find convex hull
                points_list = list(points.values())
                hull_index = cv2.convexHull(np.array(points_list), clockwise=False, returnPoints=False)
                
                # Add specific points for facial features
                add_points = [
                    [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], # Outer lips
                    [60], [61], [62], [63], [64], [65], [66], [67], # Inner lips
                    [27], [28], [29], [30], [31], [32], [33], [34], [35], # Nose
                    [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], # Eyes
                    [17], [18], [19], [20], [21], [22], [23], [24], [25], [26] # Eyebrows
                ]
                hull_index = np.concatenate((hull_index, add_points))
                
                hull = [points_list[i[0]] for i in hull_index]
                
                # Delaunay triangulation
                rect = (0, 0, img.shape[1], img.shape[0])
                dt = fbc.calculate_delaunay_triangles(rect, hull)
                
                runtime.hull = hull
                runtime.hull_index = hull_index
                runtime.dt = dt

            runtimes.append(runtime)
        return runtimes

    def render(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        output_frame = frame.copy()
        
        for idx, config in enumerate(self.filter_configs):
            runtime = self.runtimes[idx]
            points2 = landmarks # landmarks detected on frame
            
            if config['morph']:
                if runtime.dt is None or len(runtime.dt) == 0:
                    continue
                    
                hull2 = [points2[i[0]] for i in runtime.hull_index]
                
                # Use vectorized remap
                map_x, map_y, mask_remap = fbc.get_remap_maps(runtime.hull, hull2, runtime.dt, frame.shape)
                warped_img = cv2.remap(runtime.img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                
                # Process alpha mask
                mask1 = cv2.merge([mask_remap, mask_remap, mask_remap])
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                mask1 = mask1.astype(np.float32) / 255.0
                
                # Blend
                blended = warped_img.astype(np.float32) * mask1 + output_frame.astype(np.float32) * (1.0 - mask1)
                output_frame = blended.astype(np.uint8)
            else:
                # Similarity transform
                dst_points = [points2[int(list(runtime.points.keys())[0])], 
                             points2[int(list(runtime.points.keys())[1])]]
                tform = fbc.similarity_transform(list(runtime.points.values()), dst_points)
                
                trans_img = cv2.warpAffine(runtime.img, tform, (frame.shape[1], frame.shape[0]))
                trans_alpha = cv2.warpAffine(runtime.img_alpha, tform, (frame.shape[1], frame.shape[0]))
                
                mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                mask1 = mask1.astype(np.float32) / 255.0
                
                blended = trans_img.astype(np.float32) * mask1 + output_frame.astype(np.float32) * (1.0 - mask1)
                output_frame = blended.astype(np.uint8)
                
        return output_frame
