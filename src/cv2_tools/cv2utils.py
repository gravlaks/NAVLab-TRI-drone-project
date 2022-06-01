import cv2
def drawbox(img, corners):
    ptA, ptB, ptC, ptD = corners
    print(ptA)
    cv2.line(img, ptA, ptB, (0, 255, 0), 2)
    cv2.line(img, ptB, ptC, (0, 255, 0), 2)
    cv2.line(img, ptC, ptD, (0, 255, 0), 2)
    cv2.line(img, ptD, ptA, (0, 255, 0), 2)

def rescale(img, scale_percent):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return image

def fit_to_screen(img):
    h, w = img.shape[:2]

    max_height = 900.
    max_width = 2000.

    scale_percent = min(100*max_height/h, 100*max_width/w)
    img = rescale(img, scale_percent)
    return img, scale_percent


def draw_center(image, cars):
    for car in cars:
        c = car.get_center().astype(int).flatten()
        print(c)
        cv2.circle(image,c, 3, (0, 0, 255), -1)
    return image

def draw_relative(image, cars, thickness=3):
    c1 = cars[0].get_center().astype(int).flatten()
    c2 = cars[1].get_center().astype(int).flatten()

    cv2.line(image, c1, c2, color=(0, 255, 0), thickness=thickness)
    return image
def draw_detections(image, results, extra_text=""):
    
    for r in results:
        drawbox(image, r.corners)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tag_id = r.tag_id
        
        cv2.putText(image, tagFamily+" Id:"+str(tag_id), (r.corners[0][0], r.corners[0][1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        print("[INFO] tag family: {}".format(tagFamily), "tag_id", tag_id)

    cv2.putText(image, extra_text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    image, _ = fit_to_screen(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)