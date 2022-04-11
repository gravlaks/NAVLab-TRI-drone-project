

from numpy import imag
from ImageReader import ImageReader
import cv2
    
def parse_img( img, units=3):
    H, W, C = img.shape
    dx = W//(units+1)
    dy = H//(units+1)

    windows = []
    for i in range(units):
        for j in range(units):
            t = dx*i
            b = dx*(i+2)
            l = dy*(j)
            r = dy*(j+2)

            windows.append((l, r, t, b))
    return windows

if __name__=='__main__':
    folder_in = "thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run5_tandem/photos/DJI_0009/pngs/"

    image_reader = ImageReader(filepath, folder_out=folder_out)
    image_idxs = parse_img(image_reader.img)
    print(image_reader.img.shape)
    for idxs in image_idxs:
        l, r, t, b = idxs
        print(l, r, t, b)
        
        cv2.imshow("Image", image_reader.img[l:r, t:b])
        cv2.waitKey(0)