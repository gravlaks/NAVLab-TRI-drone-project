import cv2
from datetime import datetime, timedelta
class ImageReader():

    def __init__(self, filepath_in, folder_out):
        self.folder_out = folder_out
        self.raw_image = cv2.imread(filepath_in)
        self.raw_filepath = filepath_in
        start = datetime.now()

        self.convert_to_png()
        print("Conversion", datetime.now()-start)
      

        self.img = cv2.imread(self.filepath_png)

    def convert_to_png(self):
        self.filepath_png = self.folder_out  + self.raw_filepath.split("/")[-1].split(".")[-2]+".png"
        print(self.filepath_png)
        if self.raw_filepath[-3:] == "png":
            return
        
        cv2.imwrite(self.filepath_png, self.raw_image)


if __name__=='__main__':

    folder_in = "thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run5_tandem/photos/DJI_0009/pngs/"

    image_reader = ImageReader(filepath, folder_out=folder_out)