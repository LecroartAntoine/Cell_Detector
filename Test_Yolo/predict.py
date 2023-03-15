import os, cv2
from ultralytics import YOLO
from PIL import Image

print(os.getcwd())


model = YOLO("Cell_Detector/Test_Yolo/best.pt")

im1 = cv2.imread("Cell_Detector/Test_Yolo/images/train/1678464986902.jpg")



print(model.predict(im1))


# result = Image.open('runs/detect/predict/image0.jpg')


# result.show()
