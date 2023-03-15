import os, cv2
from ultralytics import YOLO
from PIL import Image


model = YOLO("C:/Users/antoine.lecroart/Documents/GitHub/Cell_Detector/Test_Yolo/best.pt")

im1 = Image.open("C:/Users/antoine.lecroart/Documents/GitHub/Cell_Detector/Test_Yolo/images/train/IMG_4906.jpg")

model.predict(im1)  # save plotted images



# result = Image.open(u'C:/Users/antoine.lecroart/Documents/GitHub/runs/detect/predict/image0.jpg')




# result.show()