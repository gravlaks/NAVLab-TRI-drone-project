import cv2
from datetime import datetime, timedelta
class ImageReader():

    def __init__(self, filepath_in, folder_out=None):
        self.folder_out = folder_out
        self.raw_image = cv2.imread(filepath_in)
        self.raw_filepath = filepath_in
        start = datetime.now()
        if self.folder_out is not None:
            self.filepath_png = self.folder_out  + self.raw_filepath.split("/")[-1].split(".")[-2]+".png"
            self.convert_to_png()
            print("Conversion", datetime.now()-start)

        else: 
            self.filepath_png = self.raw_filepath      

        self.img = cv2.imread(self.filepath_png)

    def convert_to_png(self):
        cv2.imwrite(self.filepath_png, self.raw_image)


if __name__=='__main__':

    folder_in = "thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath = folder_in + image_name +".png"
    folder_out = "thunderhill/run5_tandem/photos/DJI_0009/pngs/"

    image_reader = ImageReader(filepath)
    cv2.imshow("image", image_reader.img)
    cv2.waitKey(0)