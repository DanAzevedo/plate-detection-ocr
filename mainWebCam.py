import numpy as np
import cv2
import imutils
import pytesseract.pytesseract
import matplotlib.pyplot as plt


def findRoi(source):
    # Iremos plotar várias vezes as imagens de processamentos que iremos fazer
    # e com ela teremos controle de quando queremos plotar ou não essas imagens
    verbose = True

    video = source

    while video.isOpened():

        ret, frame = video.read()

        if (ret == False):
            break

    video = imutils.resize(video, width=500)

    if verbose == True:
        cv2.imshow('Imagem original', video)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray_img = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    if verbose == True:
        cv2.imshow('Escala de cinza', gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    bin_img = cv2.bilateralFilter(gray_img, 11, 17, 17)

    if verbose == True:
        cv2.imshow('Filtrada', bin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    canny_img = cv2.Canny(bin_img, 170, 200)

    if verbose == True:
        cv2.imshow('Bordas', canny_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contour, _ = cv2.findContours(canny_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img1 = video.copy()
    cv2.drawContours(img1, contour, -1, (0, 255, 0), 3)

    if verbose == True:
        cv2.imshow('Contornos', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contour = sorted(contour, key = cv2.contourArea, reverse = True) [:30]
    NumPlacaCnt = None

    img2 = video.copy()
    cv2.drawContours(img2, contour, -1, (0, 255, 0), 3)

    if verbose == True:
        cv2.imshow('Imagem TOP 30', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    count = 0
    idx = 1

    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            NumPlacaCnt = approx

            x, y, w, h = cv2.boundingRect(c)
            new_img = gray_img[y:y+h, x:x+w]
            cv2.imwrite('Placa' + str(idx) + '.png', new_img)
            break

    cv2.drawContours(video, [NumPlacaCnt], -1, (0, 255, 0), 3)

    if verbose == True:
        cv2.imshow('Imagem final com placa detectada', video)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cropped_img_loc = 'Placa' + str(idx) + '.png'

    if verbose == True:
        cv2.imshow('Imagem cropped placa', cv2.imread(cropped_img_loc))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    text = pytesseract.image_to_string(cropped_img_loc, config= '-l eng --oem 3 --psm 1')
    print(text)

if __name__ == "__main__":

    source = cv2.VideoCapture(0)

    findRoi(source)