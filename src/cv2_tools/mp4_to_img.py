import cv2


folder = "./thunderhill/run5_tandem/"
mov = "DJI_0009"
filepath = folder+mov+".MOV"


def video_to_mp4(input, output, fps: int = 0, frame_size: tuple = (), fourcc: str = "H264"):
    vidcap = cv2.VideoCapture(input)
    if not fps:
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    success, arr = vidcap.read()
    if not frame_size:
        height, width, _ = arr.shape
        frame_size = width, height
    writer = cv2.VideoWriter(
        output,
        apiPreference=0,
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=frame_size
    )
    while True:
        if not success:
            break
        writer.write(arr)
        success, arr = vidcap.read()
    writer.release()
    vidcap.release()

vidcap = cv2.VideoCapture(filepath)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)

    hasFrames,image = vidcap.read()
    if hasFrames:
        out_folder = folder +mov+"/image_"+ str(count)+".png"
        out_folder = folder+"/dt2e-5"+"/image_"+ str(count)+".png"
        cv2.imwrite(out_folder, image)     # save frame as JPG file
    return hasFrames
sec = 7
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
print(success)
while success:
    print(count)
    count = count + 1
    
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
    if count == 20:
        break
print("Done")