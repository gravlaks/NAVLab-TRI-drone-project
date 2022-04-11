import cv2
import numpy as np
import apriltag
from configs import *

class Measurement():
    def __init__(self, raw_img):
        self.raw_img=raw_img
        self.img = np.copy(raw_img)
        self.H, self.W, self.C = self.raw_img.shape

        options = apriltag.DetectorOptions(families=TAG_FAMILY,

                                #quad_decimate=1.0,
                                #quad_blur=0.8
                                #refine_pose=5.0
                                )
        self.detector = apriltag.Detector(options)
    def grayscale(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    def scale(self, scale_percent):
        width = int(self.W * scale_percent / 100)
        height = int(self.H * scale_percent / 100)
        dim = (width, height)
        self.img = cv2.resize(self.raw_img, dim, interpolation = cv2.INTER_AREA)
    def turn_binary(self, threshold):
        self.img = np.array(self.img>threshold, dtype=np.uint8)*255
    def detect(self):
        self.results = self.detector.detect(self.img)
        return self.results
    def reset(self):
        self.img = np.copy(self.raw_img)


if __name__=='__main__':
    folder = "thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath = folder + image_name +".jpg"

    image = cv2.imread(filepath)
    filepath_png = folder + "/"+image_name+".png"
    cv2.imwrite(filepath_png, image)

    image = cv2.imread(filepath_png)[600:2000, 500:2000]

    print(image)
    meas = Measurement(image)
    meas.grayscale()
    meas.turn_binary(threshold=200)
    res = meas.detect()
    print(res)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
 


