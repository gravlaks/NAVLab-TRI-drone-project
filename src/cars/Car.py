import numpy as np
from sift.retrieve_center import AprilTagSift
from cars.relative_distances import get_ratio
from sift.retrieve_center import get_center_w_sift, AprilTagSift
from apriltag_tools.Detector import Detector
import cv2
class Car:
    def __init__(self, tag_id, tag_family, dynamic_model, size=8.47e-2,feature_det = None):
        ## Pixel pos and velocity
        self.state = np.zeros((4, 1))
        self.trajectory = []
        self.dyn_model = dynamic_model
        self.tag_id = tag_id
        self.tag_family = tag_family
        self.sift_tag = None
        self.size=size
        self.feature_det = feature_det
    def predict(self, dt):

        self.state = self.dyn_model.f(self.state, None, dt)

    def update_state(self, detections, img, dt, low_x = 0, low_y = 0):
        det = [det for det in detections if det.tag_id == self.tag_id]
        if not len(det):
            return False
        self.detection = det[0]
        if self.sift_tag is None :
            self.sift_tag = AprilTagSift(self.detection, self.feature_det, img)


        new_c = self.detection.center.reshape((-1, 1)) + np.array([low_x, low_y]).reshape((-1, 1))
        self.set_ratio()

        self.update_trajectory(new_c, dt)
        return True

    def set_ratio(self):
        px_side = np.linalg.norm(np.array(self.detection.corners[1])-np.array(self.detection.corners[0]))
        self.ratio = self.size/px_side
    def update_state_apriltag(self, img, dt, units=1, tag_family="tag16h5", threshold=400):
        search_area, low_x, low_y = self.get_search_area(img, threshold)

        detector = Detector(img = search_area)
        detections = detector.detect(turn_binary=True, units=units, tag_family=tag_family)
        return self.update_state(detections, img, dt, low_x=low_x, low_y=low_y)

    def get_search_area(self, img, threshold):
        c = self.get_center().flatten()
        
        low_x, low_y = c.flatten() - threshold*np.ones((2, ))
        high_x, high_y = c.flatten() + threshold*np.ones((2, ))
        
        low_x = min(int(max(0, low_x)), img.shape[1])
        high_x = min(int(max(0, high_x)), img.shape[1])
        low_y = min(int(max(0, low_y)), img.shape[0])
        high_y = min(int(max(0, high_y)), img.shape[0])
        

        print(low_x, low_y, high_x, high_y, img.shape)

        if high_y-low_y<=50:
            low_y -= threshold
            high_y += threshold
            

        if high_x-low_x<=50:
            low_x -= threshold
            high_x += threshold
        low_x = min(int(max(0, low_x)), img.shape[1])
        high_x = min(int(max(0, high_x)), img.shape[1])
        low_y = min(int(max(0, low_y)), img.shape[0])
        high_y = min(int(max(0, high_y)), img.shape[0])
        search_area = img[low_y:high_y, low_x:high_x]

        try:
            assert(search_area.shape[0]>50 and search_area.shape[1] > 50), search_area.shape
        except Exception as e:
            print(e)

        return search_area, low_x, low_y
            
    def update_state_sift(self, img, dt, threshold=400):
        search_area, low_x, low_y = self.get_search_area(img, threshold)

        print(search_area.shape, low_x, low_y)
        new_c, detect = get_center_w_sift(search_area, self.sift_tag)
    
        if not detect or np.any(np.isnan(new_c)):
            cv2.imshow("frame", img)
            cv2.waitKey(0)
            return False
        else:
            assert(new_c.shape==(2,))
            new_c += np.array([low_x, low_y])

            self.update_trajectory(new_c, dt)

            return True
    def update_trajectory(self, c, dt):
        upd_state = np.vstack((
            c.reshape((-1, 1)),
            self.state[2:].reshape((-1, 1))
        ))
        if len(self.trajectory)>1:
            upd_state[2:] = (c.reshape((-1, 1))-self.state[:2])/dt
        self.trajectory.append(upd_state)
        self.state = self.trajectory[-1]


    def get_center(self):
        return self.state[:2]

    