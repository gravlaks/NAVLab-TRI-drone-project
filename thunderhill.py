import cv2
import apriltag
import numpy as np
from Detector import Detector
print(cv2.__version__)

folder = "thunderhill/run5_tandem/photos/DJI_0010/"
#images = ["image49","DJI_0001", "DJI_0002", "DJI_0003"]
#idx = 3
image_name = "image_29"
filepath_png = folder + image_name +".png"




# resize image

detector = Detector(filepath_png, None)

results = detector.detect(increase_constrast=False, adaptive_threshold=False)
image = detector.img

print("Results", len(results))

scale_percent = 50# percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)





for r in results:
    r.rescale(scale_percent)
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
    #print("Tag id", r.tag_id)
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
    tagFamily = r.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("[INFO] tag family: {}".format(tagFamily), r.tag_id)
# show the output image after AprilTag detection



cv2.imshow("Image", image)
cv2.waitKey(0)