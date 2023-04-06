import cv2 as cv
import os

def get_std(liste):

    mean = sum(liste) / len(liste)
    differences = [x - mean for x in liste]
    squared_differences = [x ** 2 for x in differences]
    mean_squared_differences = sum(squared_differences) / len(squared_differences)
    std = mean_squared_differences ** 0.5
    return(round(std, 2))

path = 'C:/Users/antoi/Documents/GitHub/Cell_Detector/Calcul Moyenne RGB/Images 28-03/Bleu'

im_R, im_G, im_B = [], [], []

for file in os.listdir(path):
    
    raw = cv.imread(path + "/" + file)
    img = cv.cvtColor(raw, cv.COLOR_BGR2RGB)
    
    pix_R, pix_G, pix_B = [], [], []
    for row in img:
        for pix in row:
            if 230 < pix[0] and 200 < pix[1]:
                pass
            else:
                pix_R.append(pix[0])
                pix_G.append(pix[1])
                pix_B.append(pix[2])
    if file[-5] != "5":
        im_R.append(round(sum(pix_R) / len(pix_R)))
        im_G.append(round(sum(pix_G) / len(pix_G)))
        im_B.append(round(sum(pix_B) / len(pix_B)))
    else:

        print(file[:-6])
        print(f"Rouge : {round(sum(im_R) / len(im_R))}")
        print(f"Rouge Ecart-Type : {get_std(im_R)}")
        print(f"Vert : {round(sum(im_G) / len(im_G))}")
        print(f"Vert Ecart-Type : {get_std(im_G)}")
        print(f"Bleu : {round(sum(im_B) / len(im_B))}")
        print(f"Bleu Ecart-Type : {get_std(im_B)}\n\n")
        im_R, im_G, im_B = [], [], []
    



