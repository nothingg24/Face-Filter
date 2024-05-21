import cv2
import numpy as np
import math

lk_params = dict(winSize=(40, 40), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.1))



class LKTracker:
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None

    def lk_track(self, next_frame, new_detected_points):
        if self.prev_frame is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_frame = next_frame
            self.prev_points = new_detected_points
            return new_detected_points
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, next_frame,
                                                                self.prev_points.astype(np.float32),
                                                                None, **lk_params)

        result = np.zeros(new_detected_points.shape)
        for i in range(0, new_detected_points.shape[0]):
            sigma = 50
            d = cv2.norm(new_detected_points[i] - new_points[i])
            alpha = math.exp(-d * d / sigma)
            point = (1 - alpha) * new_detected_points[i] + alpha * new_points[i]
            point = min(max(point[0], 0), next_frame.shape[1] - 1), min(max(point[1], 0), next_frame.shape[0] - 1)
            result[i] = point

        result = result.astype(np.float32)
        print(result.shape)

        self.prev_points = result
        self.prev_frame = next_frame.copy()
        return result


class FilterTracker():
    def __init__(self):
        self.old_frame = None
        self.previous_landmarks_set = None
        self.with_landmark = True
        self.thres = 1.0
        self.alpha = 0.95
        self.iou_thres = 0.5
        self.filter = OneEuroFilter()

    def calculate(self, now_landmarks_set):
        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0] == 0:
            self.previous_landmarks_set = now_landmarks_set
            result = now_landmarks_set
        else:
            if self.previous_landmarks_set.shape[0] == 0:
                return now_landmarks_set
            else:
              #Add the code for smooth with OneEuroFilter with no iou
                result = []
                for i in range(now_landmarks_set.shape[0]):
                    if len(now_landmarks_set[i]) == 0 or len(self.previous_landmarks_set[i]) == 0:
                        result.append(now_landmarks_set[i])
                    else:
                        result.append(self.smooth(now_landmarks_set[i], self.previous_landmarks_set[i]))
        result = np.array(result)
        self.previous_landmarks_set = result
        return result

    def smooth(self, now_landmarks, previous_landmarks):
        result = []
        for i in range(now_landmarks.shape[0]):
            dis = np.sqrt(np.square(now_landmarks[i][0] - previous_landmarks[i][0]) + np.square(
                now_landmarks[i][1] - previous_landmarks[i][1]))
            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(self.filter(now_landmarks[i], previous_landmarks[i]))
        return np.array(result)

    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)

    def __call__(self, x, x_prev):
        if x_prev is None:
            return x
        t_e = 1
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)
        self.dx_prev = dx_hat
        return x_hat


class Tracker:
    def __init__(self):
        self.filter = FilterTracker()
        self.lk_tracker = LKTracker()

    def track(self, next_frame, landmarks):
        landmarks = self.lk_tracker.lk_track(next_frame, landmarks)
        landmarks = self.filter.calculate(np.array([landmarks]))[0]
        return landmarks