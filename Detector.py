from AprilTag import AprilTag
from ImageParser import parse_img
from ImageReader import ImageReader
from Measurement import Measurement
import numpy as np
import cv2
from datetime import datetime
class Result():
    def __init__(self, res, l, r, t, b):
        self.tag_id = res.tag_id
        self.center = np.array([res.center[0]+t, res.center[1] + l])
        self.corners = res.corners
        self.tag_family = res.tag_family
        for i in range(len(self.corners)):
            self.corners[i] = np.array([self.corners[i][0]+t, self.corners[i][1] + l])

    def rescale(self, scale_percent):
        for i in range(len(self.corners)):
            self.corners[i] *= 0.01*scale_percent

        self.center *= 0.01*scale_percent
        #self.center[1] *= 0.01*scale_percent
class Detector():
    def __init__(self, filepath, folder_out, tags = [0, 2]): 
        self.tags = tags
        image_reader = ImageReader(filepath, folder_out=folder_out)
        self.img = image_reader.img
        self.image_idxs = parse_img(self.img, units=2)

    def detect(self, increase_constrast=False):
        detections = []
        tags_seen = {}

        start = datetime.now()

        for i, idxs in enumerate(self.image_idxs):
            if len(tags_seen) == 2:
                break
            l, r, t, b = idxs

            window = self.img[l:r, t:b]

            meas = Measurement(window)
            meas.grayscale()
            if increase_constrast:
                meas.imadjust()
            meas.turn_binary_adaptive()
            results = meas.detect()
            
            for result in results:
                res = Result(result, l, r, t, b)
                tag_id = res.tag_id
                
                if tag_id not in self.tags:
                    print("unseen apriltag", tag_id)
                    continue
                if tag_id in tags_seen:
                    continue
                tags_seen[tag_id] = True
                detections.append(res)
                
        print("Timing", datetime.now()-start)
        return detections


if __name__ == '__main__':
    folder_in = "thunderhill/run3/mph_10/photos/"
    image_name = "image_1"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run3/mph_10/photos/pngs/"

    detector = Detector(filepath, folder_out)
    detections = detector.detect()
    
