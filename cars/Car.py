import numpy as np
from sift.retrieve_center import AprilTagSift
from cars.relative_distances import get_ratio
from sift.retrieve_center import get_center_w_sift, AprilTagSift

class Car:
    def __init__(self, tag_id, tag_family, dynamic_model, feature_det = None):
        ## Pixel pos and velocity
        self.state = np.zeros((4, 1))
        self.trajectory = []
        self.dyn_model = dynamic_model
        self.tag_id = tag_id
        self.tag_family = tag_family
        self.sift_tag = None
        self.do_sift = True
        self.feature_det = feature_det
    def predict(self, dt):

        return self.dyn_model.f(self.state, None, dt)

    def update_state(self, detections, img=None, dt=1):
        det = [det for det in detections if det.tag_id == self.tag_id]
        if not len(det):
            return False
        self.detection = det[0]
        if self.sift_tag is None and self.do_sift:
            self.sift_tag = AprilTagSift(self.detection, self.feature_det, img)


        new_c = self.detection.center.reshape((-1, 1))
        self.ratio = get_ratio(self)

        self.update_trajectory(new_c, dt)
        return True

    def update_state_sift(self, img, dt=1):
        threshold = 500
        c = self.get_center().flatten()
        predicted_c = self.predict(dt)[:2]
        print("c", c, "pred c", predicted_c)
        low_x, low_y = predicted_c.flatten() - threshold*np.ones((2, ))
        high_x, high_y = predicted_c.flatten() + threshold*np.ones((2, ))
        low_x, high_x, low_y, high_y = int(max(low_x, 0)), int(min(high_x, img.shape[1])), int(max(low_y, 0)), int(min(high_y, img.shape[0]))

        search_area = img[low_y:high_y, low_x:high_x]
        new_c = get_center_w_sift(search_area, self.sift_tag)+ np.array([low_x, low_y])
        print("actual c", new_c)
        self.update_trajectory(new_c, dt)
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
