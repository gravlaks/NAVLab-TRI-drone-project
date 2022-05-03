import cv2
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("data/tag_comparison") if isfile(join("data/tag_comparison", f))]
print(onlyfiles)

for file in onlyfiles:
    filepath = "data/tag_comparison/"+file
    img = cv2.imread(filepath)
    cv2.imwrite("data/tag_comparison/pngs/"+"".join(file.split(".")[:-1])+".png", img)