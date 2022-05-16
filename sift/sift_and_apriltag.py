import sys
from datetime import date, datetime
# setting path
sys.path.append('../april_tags')
from os.path import exists
import numpy as np
from Detector import Detector 
from cars.Car import Car
from cars.relative_distances import get_ratio, get_relative_distance
from dynamics_models.CV import CV
import matplotlib.pyplot as plt
import cv2

def get_rel_distances(folder_in):

    idx = 10
    rel_dists = []
    det1s = []
    det2s = []
    images = []
    threshold = 70
    non_detected = []
    teslas_detected, rentals_detected, detections_cnt = 0, 0, 0
    feature_det = cv2.SIFT_create()
    car1 = Car(0, "tag36h11", CV(), feature_det=feature_det)
    car2 = Car(2, "tag36h11", CV(), feature_det=feature_det)
    warm_start=False
    apriltag_and_sift=False
    if warm_start:
        detected_1_img = cv2.imread("thunderhill/run_4/DJI_0005/image_1.png")
        detected_2_img = cv2.imread("thunderhill/run5_tandem/photos/DJI_0010/image_1.png")

        detector = Detector(img = detected_1_img)
        detections = detector.detect(turn_binary=True, units=16)
        car1.update_state(detections, detected_1_img)
        
        detector = Detector(img = detected_2_img)
        detections = detector.detect(turn_binary=True)
        car2.update_state(detections, detected_2_img)

        ratio = get_ratio(car1)

 
    while True:
        t1 = datetime.now()
        filepath = folder_in+"image_"+str(idx)+".png"
        
        if not exists(filepath) or idx == threshold:
            #print(filepath, "does not exist. Quitting")
            #print("Or top threshold,  ", threshold, " reached")
            break
        img = cv2.imread(filepath)
        detect1, detect2 = False, False
        if apriltag_and_sift or (car1.sift_tag is None or car2.sift_tag is None):
            detector = Detector(img=img)
            detections = detector.detect(turn_binary=True)

            detect1 = car1.update_state(detections, img)
            detect2 = car2.update_state(detections, img)

            if detect1: teslas_detected+=1
            if detect2: rentals_detected+=1

        if detect1 and detect2:
            print("apriltag detection only")
            detections_cnt+=1
            det1s.append(car1.detection)
            det2s.append(car2.detection)

            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
            images.append(detector.img)
        else:
            print("Apriltag+sift")
            if car1.sift_tag is None or car2.sift_tag is None: 
                continue
            #img_blur = cv2.blur(img, ksize=(2,2))
            if not detect1:
                car1.update_state_sift(img)
            if not detect2:
                car2.update_state_sift(img)
           
            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
            images.append(img)
            #print("tesla: ", len(det1),"rental: ", len(det2))
 
            non_detected.append(filepath)
        idx+=1
        t2 = datetime.now()
        print(t2-t1)
    # print("One: ", one_detections, "avg: ", one_detections/idx)
    print("Detection rate: ", detections_cnt, "avg: ", (detections_cnt)/(idx-1))
    print("Teslas: ", teslas_detected, "avg: ", teslas_detected/(idx-1))
    print("Rentals: ", rentals_detected, "avg: ", rentals_detected/(idx-1))
    print(f"& {teslas_detected/(idx-1)} & {rentals_detected/(idx-1)} & {(detections_cnt)/(idx-1)}\\\\")
    rel_dists = np.array(rel_dists)
    det1s = np.array(det1s)
    det2s = np.array(det2s)

    return rel_dists, det1s, det2s, images

def plot_relative_distances(rel_dists, filepath):

    fig = plt.figure()
    plt.plot(rel_dists[:, 0], rel_dists[:, 1])
    plt.savefig(filepath, dpi=fig.dpi)
    plt.show()

def plot_locations(pos1, pos2):
    plt.plot(pos1[:, 0], pos1[:, 1])
    plt.plot(pos2[:, 0], pos2[:, 1])
    plt.show()




    
if __name__ == '__main__':
    folder_in = "thunderhill/run5_tandem/photos/DJI_0009/"
    assert(folder_in[-1] == "/")
    rel_dist_filepath = "thunderhill/plots/run5.png"
    
    rel_dists, det1s, det2s, images = get_rel_distances(folder_in=folder_in)
  
    plot_relative_distances(rel_dists, filepath=rel_dist_filepath)
    #plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    #plot_imgs(images, det1s, det2s)