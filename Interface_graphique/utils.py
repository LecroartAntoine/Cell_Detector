import cv2, os
import numpy as np
from ultralytics import YOLO
    
def yolo_detection(image, confiance):

    model = YOLO(os.path.dirname(os.path.abspath(__file__)) + '/Assets/model.pt')

    pred = model.predict(image, conf = confiance, max_det = 2000, iou = 0.3)

    return pred[0]


def plot_bboxes(img, boxes, thickness, score=True):
  
    image = img.copy()

    colors = [(255, 0, 0)]
  

    for box in boxes:
        
        if score :
            label = str(round(100 * float(box[-2]), 1)) + "%"

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

def show_all_detections(image):

    try:
        liste_images = []
        nb_image = len(image['pred'].boxes.boxes)
        fill = np.zeros((50, 50, 3), dtype = "uint8")

        for box in image['pred'].boxes.boxes:

            cell = image['image'][int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cell = cv2.resize(cell, (50, 50), interpolation=cv2.INTER_AREA)
            cell_bw = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)

            row_black_avg = []
            for row in cell_bw :
                row_values = []
                for pixel in row:
                    if pixel < 200:
                        row_values.append(pixel)
                if row_values:
                    row_black_avg.append(sum(row_values) / len(row_values))
            if row_black_avg:
                black_avg = round(sum(row_black_avg) / len(row_black_avg))

            liste_images.append([black_avg, cell])

        while int(nb_image**0.5)**2 != nb_image:
            liste_images.append([0, fill])
            nb_image += 1

        liste_images = [image[1] for image in sorted(liste_images, key=lambda x: x[0], reverse=True)]

        liste_images_h = []
        for i in range(0, nb_image, int(nb_image**0.5)):

            liste_images_h.append(cv2.hconcat(liste_images[i:i+int(nb_image**0.5)]))

        while np.array_equal(liste_images_h[-1], np.zeros((50, 50 * int(nb_image**0.5), 3), dtype = "uint8")):
            liste_images_h.pop(-1)
            
        image_combin = cv2.vconcat(liste_images_h)

    except:
        image_combin = np.zeros((200, 200, 3), dtype = "uint8")

    return image_combin

def calcul_malassez(images, img_40):

    if img_40:
        w, h = 230, 185
    else:
        w, h = 580, 462
    for key in images:
        liste_nb_cells = []
        liste_detection = [box.tolist() for box in images[key]['pred'].boxes.boxes]

        for y in range(images[key]['image'].shape[0]//h):
            for x in range(images[key]['image'].shape[1]//w):
                x_min = x * w
                x_max = x * w + w
                y_min = y * h
                y_max = y * h + h
                nb_cells = 0


                for box in liste_detection[:]:
                    
                    if box[0] > x_min and box[0] < x_max and box[3] > y_min and box[3] < y_max:
                        nb_cells += 1
                        liste_detection.remove(box)

                liste_nb_cells.append(nb_cells)

        images[key]['concentration'] = round(sum(liste_nb_cells) / (len(liste_nb_cells)), 3)

    return (images)

def get_std(liste):

    mean = sum(liste) / len(liste)
    differences = [x - mean for x in liste]
    squared_differences = [x ** 2 for x in differences]
    mean_squared_differences = sum(squared_differences) / len(squared_differences)
    std = mean_squared_differences ** 0.5
    return(round(std, 2))