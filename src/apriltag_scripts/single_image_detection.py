import cv2
import apriltag
import numpy as np
import sys
sys.path.append('../src')
from apriltag_tools.Detector import Detector
from cv2_tools.cv2utils import *
print(cv2.__version__)

folder = "thunderhill/run_4/DJI_0004/"

image_name = "image_29"
filepath_png = folder + image_name +".png"





# resize image

detector = Detector(filepath_png, None)

results = detector.detect(turn_binary=True, units=8)
image = detector.img

print("Results", len(results))





draw_detections(image, results)

