import os, cv2
from ultralytics import YOLO
from PIL import Image


model = YOLO("C:/Users/antoine.lecroart/Documents/GitHub/Cell_Detector/Test_Yolo/best.pt")

im1 = Image.open("C:/Users/antoine.lecroart/Documents/GitHub/Cell_Detector/Test_Yolo/images/train/IMG_4892.jpg")

results = model.predict(source=im1, save=True)  # save plotted images

print(results[0].boxes)


result = Image.open(u'C:/Users/antoine.lecroart/Documents/GitHub/runs/detect/predict/image0.jpg')

# cv2.rectangle(result,(x_min,y_min),(x_max,y_max),color,2)


result.show()