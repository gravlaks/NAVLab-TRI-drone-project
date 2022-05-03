import json
from configs import * 
from Detector import Detector
import numpy as np
def get_ratio(detections):
    """
    ratio is m/px
    """
    det = detections[1]
    tag_id = det.tag_id

    testing_day="thunderhill_04_07"
    filepath = "data/" + "measurements.json"
    with open(filepath, "r") as f:
        car = json.load(f)[testing_day][str(tag_id)]
        w = car["w"]
        h = car["h"]
        type = car["type"]

    if tag_id == 0:

        ###OBS: have to check this
        pixel_width = np.linalg.norm(det.corners[0]-det.corners[1])
        pixel_height = np.linalg.norm(det.corners[1]-det.corners[2])
    if tag_id == 2:

        ###OBS: have to check this
        pixel_width = np.linalg.norm(det.corners[0]-det.corners[1])
        pixel_height = np.linalg.norm(det.corners[1]-det.corners[2])

    ratio1 = w/pixel_width
    return ratio1

def get_relative_distance(detections):
    ratio = get_ratio(detections)
    det1 = [det for det in detections if det.tag_id == 2][0]
    det2 = [det for det in detections if det.tag_id == 0][0] #tesla

    relative_distance_px = det2.center - det1.center
    relative_distance = ratio*relative_distance_px
    return relative_distance




if __name__=='__main__':
    folder_in = "thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run5_tandem/photos/DJI_0009/pngs/"

    detector = Detector(filepath, folder_out)
    detections = detector.detect()
    print(get_relative_distance(detections))