import cv2
import sys
  
# setting path
sys.path.append('../april_tags')
import cv2utils
from sift.SIFT_detector import get_matches_lowe_ratio
from rotations_2d.angle_between import angle_between, get_rotation_matrix
import numpy as np
from Detector import Detector
import matplotlib.pyplot as plt

folder = "thunderhill/run5_tandem/photos/DJI_0009/"
sift = cv2.SIFT_create()

filepath_1 = "image_3.png"
filepath_2 = "image_17.png"

img1 = cv2utils.rescale(cv2.imread(folder+filepath_1), 100)
#img1 = cv2utils.fit_to_screen(img1)

img2 = cv2utils.rescale(cv2.imread(folder+filepath_2), 100)


### Save first keypoints
detector = Detector(img=img1)
results = detector.detect(increase_constrast=False, 
                    adaptive_threshold=False, turn_binary=True,
                    tag_family="tag36h11")
print(len(results))
print(results[0].corners)
corners = results[0].corners
corners = np.array(corners).reshape((-1, 2))
center = results[0].center
print("center on img1", center)
low_x, low_y = np.min(corners, axis=0)
high_x, high_y = np.max(corners, axis=0)
print(low_x, low_y, high_x, high_y)
search_area = img1[low_y:high_y, low_x:high_x]

gray= cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
kps_img1, des1 = sift.detectAndCompute(gray,None)

relatives = [np.array(kp.pt)+np.array([low_x, low_y])-center for kp in kps_img1]

print(len(kps_img1))



### Match with other image

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kps_img2, des2 = sift.detectAndCompute(gray2,None)
good = get_matches_lowe_ratio(des1, des2, 0.75, True, gray, gray2, kps_img1, kps_img2)

def get_angle(good):
    angles = []
    for i in range(0, len(good)-1, 2):
        match0 = good[i][0]
        match1 = good[len(good)-i-1][0]
        vec1 = np.array(kps_img2[match0.trainIdx].pt)-np.array(kps_img2[match1.trainIdx].pt)
        vec0 = np.array(kps_img1[match0.queryIdx].pt)-np.array(kps_img1[match1.queryIdx].pt)

        #Rotation from first image to second image
        angle = angle_between(vec0, vec1)
        angles.append(angle)
    return angles
    
angles = get_angle(good)
detector = Detector(img=img2)
results = detector.detect(increase_constrast=False, 
                    adaptive_threshold=False, turn_binary=True,
                    tag_family="tag36h11")
def get_center(good, angles):
    for match, angle in zip(good, angles):
        R = get_rotation_matrix(angle)

        match0 = match[0]
        idx = match0.trainIdx
        print(idx)
        rel = relatives[match0.queryIdx]
        vec1 = np.array(kps_img2[match0.trainIdx].pt)
        rel_rot = R@rel
        c = vec1-rel_rot
        plt.quiver(0,0,rel[0], rel[1], label="rel", color="red")
        plt.quiver(0,0,rel_rot[0], rel_rot[1], label="rel rot", color="green")
        plt.scatter([vec1[0]-results[0].center[0]], [vec1[1]-results[0].center[0]])
        plt.legend()
        #plt.plot(0, 0,rel_rot[0], rel_rot[1], label="rel rot")
        plt.show()
        print("rel", rel, "rel rot",rel_rot)
        print("c", c, "vec1", vec1, "angle", angle)
        #Rotation from first image to second image
print("img2 centers", results[0].center, results[1].center)

get_center(good, angles)
print(len(good))


