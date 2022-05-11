from AprilTag import AprilTag
from ImageParser import parse_img
from ImageReader import ImageReader
from Measurement import Measurement
import numpy as np
import cv2
#from PyTorchYOLOv3.pytorchyolo.detect import detect
from cv2utils import drawbox
from datetime import datetime
#from PyTorchYOLOv3.pytorchyolo import detect, models
# model = models.load_model(
#   "PyTorchYOLOv3/config/yolov3.cfg", 
#   "PyTorchYOLOv3/weights/yolov3.weights")
class Result():
    def __init__(self, res, l, r, t, b):
        self.tag_id = res.tag_id
        self.center = np.array([res.center[0]+l, res.center[1] + t], dtype=np.int16)
        self.corners = res.corners
        self.tag_family = res.tag_family
        for i in range(len(self.corners)):
            self.corners[i] = [self.corners[i][0]+l, self.corners[i][1] + t]
        self.corners = np.array(self.corners, dtype=np.int16)
    def rescale(self, scale_percent):
        for i in range(len(self.corners)):
            self.corners[i] = np.array(self.corners[i], dtype=np.float16)*0.01*scale_percent
        self.corners = np.array(self.corners, dtype=np.int16)
        self.center = np.array(self.center, dtype=np.float16)*0.01*scale_percent
        self.center = np.array(self.center, dtype=np.int16)

        #self.center[1] *= 0.01*scale_percent
class Detector():
    def __init__(self, filepath=None, img=None, tags = [i for i in range(30)]): 
        self.tags = tags


        if filepath is not None:
            self.img = cv2.imread(filepath)
        else:
            self.img = img

    def detect(self, increase_constrast=False, adaptive_threshold=True, tag_family = "tag36h11", turn_binary=True):
        detections = []
        tags_seen = {}
        visualize = False
        self.image_idxs = parse_img(self.img, units=4)


        start = datetime.now()

        for i, idxs in enumerate(self.image_idxs):
            if len(tags_seen) == 10:
                break
            l, r, t, b = idxs
            window = self.img[t:b, l:r]
            #boxes = detect.detect_image(model, window, conf_thres=0.1)
            #for box in boxes:
                #x1, y1, x2, y2, conf, class_ = box
                #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #l, r, t, b = l+x1, l+x2, t+y1, t+y2 


                #new_window = window[y1:y2,x1:x2]
            meas = Measurement(window, tag_family=tag_family)
            meas.grayscale()
            if increase_constrast:
                raise NotImplemented
            if turn_binary:
                if adaptive_threshold:
                    meas.turn_binary_adaptive()
                elif False:
                    meas.turn_binary()
                elif True:
                    meas.turn_binary_const(220)
            results = meas.detect()
            
            for result in results:
                res = Result(result, l, r, t, b)
                tag_id = res.tag_id
                
                if tag_id not in self.tags:
                    #print("unseen apriltag", tag_id)
                    continue
                if tag_id in tags_seen:
                    continue
                tags_seen[tag_id] = True
                detections.append(res)
            if visualize:
                pass
                #cv2.imshow(f"Image", meas.img)
                #cv2.waitKey(0)
        #print("Timing", datetime.now()-start)
        return detections


if __name__ == '__main__':
    folder_in = "thunderhill/run3/mph_10/photos/"
    image_name = "image_1"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run3/mph_10/photos/pngs/"

    detector = Detector(filepath, folder_out)
    detections = detector.detect()
    print(detections)
    
