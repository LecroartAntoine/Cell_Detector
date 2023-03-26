import cv2, os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO(os.path.dirname(os.path.abspath(__file__)) + '/Assets/model.pt')

image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/1678464987032.jpg', cv2.IMREAD_UNCHANGED)


pred = model(image, conf = 0.4)

liste_images = []
liste_images_h = []
nb_image = len(pred[0].boxes.boxes)
fill = np.zeros((50, 50, 3), dtype = "uint8")


for box in pred[0].boxes.boxes:

    cell = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    cell = cv2.resize(cell, (50, 50), interpolation=cv2.INTER_AREA)
    liste_images.append(cell)

while int(nb_image**0.5)**2 != nb_image:
    liste_images.append(fill)
    nb_image += 1

for i in range(0, nb_image, int(nb_image**0.5)):

    liste_images_h.append(cv2.hconcat(liste_images[i:i+int(nb_image**0.5)]))

if liste_images_h[-1].all() == np.zeros((50, 50*int(nb_image**0.5), 3), dtype = "uint8").all():
    image_combin = cv2.vconcat(liste_images_h[:-1])

else:
    image_combin = cv2.vconcat(liste_images_h)



