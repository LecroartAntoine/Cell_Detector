import cv2, os
import numpy as np
from ultralytics import YOLO

def cropAuto(img, thresh):
    src = img.copy()

    # On ne garde que la dimension 1 (= Vert)
    src[:,:,2] = np.zeros([src.shape[0], src.shape[1]])
    src[:,:,0] = np.zeros([src.shape[0], src.shape[1]])

    # Détection de contours
    gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    _,edge = cv2.threshold(gray, thresh, 1, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # On ne garde que le plus gros contour trouvé
        nb_cont = 0
        best = -1
        for i, contour in enumerate(contours):
            if len(contours[i]) > nb_cont:
                nb_cont = len(contours[i])
                best = i

        # On trouve les dimensions du contour
        xmin, xmax, ymin, ymax = 5000, 0, 5000, 0
        for i, contour in enumerate(contours[best]):
            if contour[0][0] > xmax:
                xmax = contour[0][0] 

            if contour[0][0] < xmin:
                xmin = contour[0][0] 

            if contour[0][1] > ymax:
                ymax = contour[0][1] 

            if contour[0][1] < ymin:
                ymin = contour[0][1]

        # Découpage de l'image avec les dimensions trouvées
        cropped = img[ymin:ymax, xmin:xmax]

        # Redimension de l'image par un multiple de 32
        x = cropped.shape[0] if cropped.shape[0] > cropped.shape[1] else cropped.shape[1]
        while x % 32 != 0:
            x -= 1
        resized = cv2.resize(cropped, (x, x), interpolation = cv2.INTER_AREA)

        return resized
    except:
        return img
    
def yolo_detection(image, confiance):

    model = YOLO(os.path.dirname(os.path.abspath(__file__)) + '/Assets/model.pt')

    pred = model(image, conf = confiance)

    return pred[0]


def plot_bboxes(img, boxes, thickness, score=True, name=True):
  
    image = img.copy()
    labels = {0: "clean", 1: "marked"}

    colors = [(0, 255, 0),(255, 0, 0)]
  

    for box in boxes:
        label = ''

        if name:
            label = labels[int(box[-1])]

        if score :
            label = label + " " + str(round(100 * float(box[-2]), 1)) + "%"

        color = colors[int(box[-1])]

        image = box_label(image, box, label, color, thickness)

    return image


def box_label(image, box, label, color, thickness, txt_color=(255, 255, 255)):
    
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)

    if label:
        tf = max(thickness- 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image,
                    label, 
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    thickness / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    
    return image

def show_all_detections(image, pred):

    liste_images = []
    liste_images_h = []
    nb_image = len(pred.boxes.boxes)
    fill = np.zeros((50, 50, 3), dtype = "uint8")


    for box in pred.boxes.boxes:

        cell = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cell = cv2.resize(cell, (50, 50), interpolation=cv2.INTER_AREA)
        liste_images.append(cell)

    while int(nb_image**0.5)**2 != nb_image:
        liste_images.append(fill)
        nb_image += 1

    for i in range(0, nb_image, int(nb_image**0.5)):

        liste_images_h.append(cv2.hconcat(liste_images[i:i+int(nb_image**0.5)]))

    while np.array_equal(liste_images_h[-1], np.zeros((50, 50 * int(nb_image**0.5), 3), dtype = "uint8")):
        liste_images_h.pop(-1)
        
    image_combin = cv2.vconcat(liste_images_h)

    return image_combin