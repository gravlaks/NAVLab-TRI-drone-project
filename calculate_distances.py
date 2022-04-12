from Detector import Detector

from os.path import exists
import numpy as np
from relative_distances import get_relative_distance
import matplotlib.pyplot as plt
import cv2
def get_rel_distances(folder_in, folder_out, png=False):

    idx = 1
    rel_dists = []
    det1s = []
    det2s = []
    images = []
    threshold = 100
    non_detected = []
    while True:
        if png:
            filepath = folder_in+"image_"+str(idx)+".png"
        else:
            filepath = folder_in+"image_"+str(idx)+".jpg"
        if not exists(filepath) or idx == threshold:
            print(filepath, "does not exist. Quitting")
            print("Or top threshold,  ", threshold, " reached")
            break
        
        detector = Detector(filepath, folder_out)
        detections = detector.detect()
        det1 = [det for det in detections if det.tag_id == 0]
        det2 = [det for det in detections if det.tag_id == 2]

        if len(det1) and len(det2):
            det1s.append(det1[0])
            det2s.append(det2[0])

            rel_dist = get_relative_distance(detections)
            rel_dists.append(rel_dist)
            images.append(detector.img)

        else:
            print("non detection", filepath)
            print(len(det1), len(det2))
            non_detected.append(filepath)
        idx+=1
    rel_dists = np.array(rel_dists)
    det1s = np.array(det1s)
    det2s = np.array(det2s)

    return rel_dists, det1s, det2s, images

def plot_relative_distances(rel_dists):
    plt.plot(rel_dists[:, 0], rel_dists[:, 1])
    plt.show()

def plot_locations(pos1, pos2):
    plt.plot(pos1[:, 0], pos1[:, 1])
    plt.plot(pos2[:, 0], pos2[:, 1])
    plt.show()

def plot_imgs(images, det1s, det2s):
    for det1, det2, img in zip(det1s, det2s,  images):
        plot_result(img, det1, det2)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

def plot_result(image, det1, det2):
    
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
    #print("Tag id", r.tag_id)
    dets = [det1, det2]
    for r in dets:
        (ptA, ptB, ptC, ptD) = r.corners
        #print("Hamming", r.hamming)
        #print("Homography", r.homography)
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 1, (0, 0, 255), -1)
        # draw the tag family on the image
    
if __name__ == '__main__':
    folder_in = "thunderhill/run_4/DJI_0007/"
    #image_name = "image_14"
    #filepath = folder_in + image_name +".jpg"
    folder_out = folder_in+"pngs/"

    rel_dists, det1s, det2s, images = get_rel_distances(folder_in=folder_in, folder_out=folder_in, png=True)
    plot_relative_distances(rel_dists)
    plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    plot_imgs(images, det1s, det2s)