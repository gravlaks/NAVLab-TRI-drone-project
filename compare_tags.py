from unittest import result
import cv2
import apriltag
import numpy as np
from Detector import Detector
from cv2utils import draw_detections, rescale

from os import listdir
from os.path import isfile, join
folder = "data/tag_comparison/pngs"
dir_files= [f for f in listdir(folder) if isfile(join(folder, f))]
filepaths = [folder + "/"+file for file in dir_files]


tag_families = ["tag36h11", "tag16h5"]
scale_percents = [100]
for filepath_png in filepaths:

    for scale_percent in scale_percents:
        for tag_family in tag_families:
            image = cv2.imread(filepath_png)
            image = rescale(image, scale_percent )
            print(image.shape)
            detector = Detector(img=image)

            results = detector.detect(increase_constrast=False, 
                    adaptive_threshold=False, turn_binary=False,
                    tag_family=tag_family)
            image = detector.img

            print("Results", tag_family, scale_percent, len(results))
            if len(results):
                print(results[0].corners)

            extra_text = f"{tag_family}, {scale_percent}"
            draw_detections(image, results, extra_text)

