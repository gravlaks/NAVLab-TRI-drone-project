import numpy as np
import json
def get_ratio(car):
    """
    ratio is m/px
    """
    return 1
    det = car.detection

    testing_day="thunderhill_04_07"
    filepath = "data/" + "measurements.json"
    with open(filepath, "r") as f:
        car_json = json.load(f)[testing_day][str(car.tag_id)]
        w = car_json["w"]
        h = car_json["h"]
        type = car_json["type"]

    if car.tag_id == 0:

        ###OBS: have to check this
        pixel_width = np.linalg.norm(det.corners[0]-det.corners[1])
        pixel_height = np.linalg.norm(det.corners[1]-det.corners[2])
    if car.tag_id == 2:

        ###OBS: have to check this
        pixel_width = np.linalg.norm(det.corners[0]-det.corners[1])
        pixel_height = np.linalg.norm(det.corners[1]-det.corners[2])

    ratio1 = w/pixel_width
    return 1
    return ratio1


def get_relative_distance(car1, car2):
    ratio = get_ratio(car1)


    relative_distance_px = car2.get_center() - car1.get_center()
    relative_distance = ratio*relative_distance_px
    return relative_distance.reshape((-1, 1))