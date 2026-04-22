import cv2
import numpy as np
import math

lk_params = dict(winSize=(40, 40), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.1))

def dist(a, b):
    return np.sum(np.power(a - b, 2), 1)

class LKTracker:
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None

    def delta_fn(self, prev_points, new_detected, lk_tracked):
        result = np.zeros(new_detected.shape)
        dist_detect = dist(new_detected, prev_points)
        dist_lk = dist(new_detected, lk_tracked)
        
        eye_indices = np.array(list(set(range(36, 48))))
        rest_indices = np.array(list(set(range(68)) - set(range(36, 48))))

        # Logic: If detector moved a lot, trust detector. If moved little, trust LK.
        # This helps with small tremors.
        
        # Eyes
        thres_eye = 1.5
        w_lk_small = 0.90 # Trust LK more for small eye movements
        w_det_large = 0.95 # Trust Detector for large movements
        
        d_eye = dist_detect[eye_indices]
        res_eye = result[eye_indices]
        
        mask_small = d_eye < thres_eye
        res_eye[mask_small] = lk_tracked[eye_indices][mask_small] * w_lk_small + new_detected[eye_indices][mask_small] * (1 - w_lk_small)
        res_eye[~mask_small] = new_detected[eye_indices][~mask_small] * w_det_large + lk_tracked[eye_indices][~mask_small] * (1 - w_det_large)
        result[eye_indices] = res_eye

        # Rest of face
        thres_rest = 5.0
        d_rest = dist_detect[rest_indices]
        res_rest = result[rest_indices]
        
        mask_small_rest = d_rest < thres_rest
        res_rest[mask_small_rest] = lk_tracked[rest_indices][mask_small_rest] * 0.85 + new_detected[rest_indices][mask_small_rest] * 0.15
        res_rest[~mask_small_rest] = new_detected[rest_indices][~mask_small_rest] * 0.90 + lk_tracked[rest_indices][~mask_small_rest] * 0.10
        result[rest_indices] = res_rest
        
        return result

    def lk_track(self, next_frame, new_detected_points):
        if self.prev_frame is None:
            self.prev_frame = next_frame.copy()
            self.prev_points = new_detected_points
            return new_detected_points
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, next_frame,
                                                                self.prev_points.astype(np.float32),
                                                                None, **lk_params)
        
        result = self.delta_fn(self.prev_points, new_detected_points, new_points)
        self.prev_points = result
        self.prev_frame = next_frame.copy()
        return result

class OneEuroFilter:
    def __init__(self, min_cutoff=0.1, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
            
        t_e = 1.0 # Assume constant 1/fps
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class Tracker:
    def __init__(self):
        self.lk_tracker = LKTracker()
        # Create a OneEuroFilter for each of the 68 landmarks (each having x, y)
        self.filters = [OneEuroFilter(min_cutoff=0.05, beta=0.005) for _ in range(68 * 2)]

    def track(self, next_frame, landmarks):
        # 1. LK Tracking + Detection Fusion
        landmarks = self.lk_tracker.lk_track(next_frame, landmarks)
        
        # 2. OneEuroFilter for each coordinate
        flat_landmarks = landmarks.flatten()
        filtered_landmarks = np.zeros_like(flat_landmarks)
        for i in range(len(flat_landmarks)):
            filtered_landmarks[i] = self.filters[i](flat_landmarks[i])
            
        return filtered_landmarks.reshape(68, 2)