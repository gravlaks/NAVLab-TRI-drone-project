import cv2
import apriltag
import numpy as np
print(cv2.__version__)

folder = "thunderhill/run_4/DJI_0004/"
#images = ["image49","DJI_0001", "DJI_0002", "DJI_0003"]
#idx = 3
image_name = "image_23"
filepath_png = folder + image_name +".png"



image = cv2.imread(filepath_png)[400:600,2400:2600]

image2 = cv2.imread(filepath_png)[:2000,1900:2400]

h,w, _ = image.shape

print(image.shape)
scale_percent = 100# percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.array(gray>200, dtype=np.uint8)*255
print(gray)

options = apriltag.DetectorOptions(families="tag36h11",

                                #quad_decimate=1.0,
                                #quad_blur=0.8
                                refine_pose=5.0
                                )
detector = apriltag.Detector(options)
results = detector.detect(gray)

print("Results", len(results))
if image2 is not None:

    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    results2 = detector.detect(gray2)
    print("Results2", len(results2))




for r in results:
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
    #print("Tag id", r.tag_id)
    print(r.center)
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
    print("[INFO] tag family: {}".format(tagFamily))
# show the output image after AprilTag detection



cv2.imshow("Image", image)
cv2.waitKey(0)