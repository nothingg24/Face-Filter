import cv2
import numpy as np
import math

class KalmanFilter: #(cv2.KalmanFilter)
    def __init__(self, point) -> None:
        self.point = point
        # self.point = list(point)
        self.stateNum = 4
        self.measureNum = 2
        self.kalman = cv2.KalmanFilter(self.stateNum, self.measureNum, 0, cv2.CV_64F)
        self.is_predicted = False
        self.deltaTime = 0.2 #0.2/1/0.5/0.7
        self.accelNoiseMag = 0.3 #0.3/1/0.7/1
        self.init()

    def init(self):
        self.kalman.transitionMatrix = np.array([[1, 0, self.deltaTime, 0],
                                                [0, 1, 0, self.deltaTime],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float64)
        self.kalman.statePre = np.array([[self.point[0]], [self.point[1]], [1], [1]], dtype=np.float64)
        self.kalman.statePost = np.array([[self.point[0]], [self.point[1]], [0], [0]], dtype=np.float64)

        cv2.setIdentity(self.kalman.measurementMatrix)

        self.kalman.processNoiseCov = np.array([[pow(self.deltaTime, 4.0) / 4.0, 0, pow(self.deltaTime, 3.0) / 2.0, 0],
                                                [0, pow(self.deltaTime, 4.0) / 4.0, 0, pow(self.deltaTime, 3.0) / 2.0],
                                                [pow(self.deltaTime, 3.0) / 2.0, 0, pow(self.deltaTime, 2.0), 0],
                                                [0, pow(self.deltaTime, 3.0) / 2.0, 0, pow(self.deltaTime, 2.0)]], dtype=np.float64) * self.accelNoiseMag
        
        # cv2.setIdentity(self.kalman.processNoiseCov, 0.001)
        cv2.setIdentity(self.kalman.measurementNoiseCov, 0.1)#0.1
        cv2.setIdentity(self.kalman.errorCovPost, 0.1)

    def update(self, point):
        measurement = np.array([[np.float64(point[0])], [np.float64(point[1])]], dtype=np.float64)

        if point[0] < 0 or point[1] < 0:
            self.predict()
            measurement[0, 0] = self.point[0]
            measurement[1, 0] = self.point[1]
            self.is_predicted = True
        else:
            measurement[0, 0] = point[0]
            measurement[1, 0] = point[1]
            self.is_predicted = False

        estimated = self.kalman.correct(measurement)
        self.point[0] = estimated[0, 0]
        self.point[1] = estimated[1, 0]

        self.predict()

    def predict(self):
        prediction = self.kalman.predict()
        self.point[0] = prediction[0, 0]
        self.point[1] = prediction[1, 0]
        return self.point
    
    def getPoint(self):
        return self.point
    
    def isPredicted(self):
        return self.is_predicted
    
    @staticmethod
    def trackpoints(prevFrame, currFrame, currLandmarks, trackPoints):
        # lk_params = dict(winSize=(7, 7), maxLevel=3,
        #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), flags=0, minEigThreshold=0.001)
        # lk_params = dict(winSize=(101, 101), maxLevel=15,
        #                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), flags=0, minEigThreshold=0.001)
        prevLandmarks = [pts.getPoint() for pts in trackPoints]
        # prevLandmarks = trackPoints

        newLandmarks, status, err = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prevLandmarks,#np.array(prevLandmarks, dtype=np.float32)/prevLandmarks
                                                             np.array(currLandmarks, dtype=np.float64), **lk_params) #np.array(currLandmarks, dtype=np.float64)/None
        
        for i in range(len(status)):
            if status[i]:
                sigma = 50
                d = cv2.norm(np.array(currLandmarks[i]) - newLandmarks[i])
                alpha = math.exp(-d * d / sigma)
                point = (1 - alpha) * np.array(currLandmarks[i]) + alpha * newLandmarks[i]
                point = min(max(point[0], 0), currFrame.shape[1] - 1), min(max(point[1], 0), currFrame.shape[0] - 1)

                trackPoints[i].update(point) #(newLandmarks[i]+currLandmarks[i])/2 or point or KalmanFilter(trackPoints[i]).update(point)
            else:
                trackPoints[i].update(currLandmarks[i]) #KalmanFilter(trackPoints[i]).update(currLandmarks[i])
        



        


