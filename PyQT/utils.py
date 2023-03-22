import cv2
import numpy as np

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