import pytesseract
import cv2
from datetime import datetime

def drawContours(contours, img):
    for c in contours:
        # perimetro do contorno, verifica se o contorno é fechado
        perimeter = cv2.arcLength(c, True)# Qtdd de vértices do contorno e mostra se o contorno é fechado
        if perimeter > 120:# Elimino formar menores da imagem e deixo retângulos grandes
            # aproxima os contornos da forma correspondente
            approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)# Une contornos com apoximação
            # verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
            if len(approx) == 4:
                # Contorna a placa atraves dos contornos encontrados
                (x, y, lar, alt) = cv2.boundingRect(c)# Aproximo ela de um retângulo
                cv2.rectangle(img, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                # segmenta a placa da imagem
                roi = img[y:y + alt, x:x + lar]
                cv2.imwrite("Imagens/roi.png", roi)


def findRectangle(source):
    # Captura ou Video
    video = cv2.VideoCapture(source)

    while video.isOpened():

        ret, frame = video.read()

        if (ret == False):
            break

        # area de localização
        area = frame[500:, 300:800]

        # escala de cinza
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # Binarização
        ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

        # desfoque
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        # lista os contornos
        contours, hierarchy = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # limite horizontal
        cv2.line(frame, (0, 500), (1280, 500), (0, 0, 255), 1)
        # limite vertical 1
        cv2.line(frame, (300, 0), (300, 720), (0, 0, 255), 1)
        # limite vertical 2
        cv2.line(frame, (800, 0), (800, 720), (0, 0, 255), 1)

        # cv2.imshow('FRAME', frame)

        drawContours(contours, area)

        cv2.imshow('RES', area)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    video.release()
    preProcessingRoi()
    cv2.destroyAllWindows()


def preProcessingRoi():
    img_roi = cv2.imread("Imagens/roi.png")
    # cv2.imshow("ENTRADA", img_roi)
    if img_roi is None:
        return

    # redmensiona a imagem da placa em 6x
    img = cv2.resize(img_roi, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Escala Cinza", img)

    # Binariza imagem
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Binária", img)

    # Desfoque na Imagem
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Desfoque", img)

    cv2.imwrite("Imagens/roi-ocr.png", img)

    return img


def recognitionOCR():
    img_roi_ocr = cv2.imread("Imagens/roi-ocr.png")
    # Aplica reconhecimento OCR no ROI com o Tesseract
    img_roi_ocr = cv2.cvtColor(img_roi_ocr, cv2.COLOR_BGR2RGB)
    if img_roi_ocr is None:
        return

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    output = pytesseract.image_to_string(img_roi_ocr, lang='eng')
    date = datetime.now()
    date_hour = date.strftime('%d/%m/%Y %H:%M')
    print('===========================')
    print(f'Data/Hora: ', date_hour)
    print(f'Placa: ', output)
    print('===========================')

if __name__ == "__main__":

    source = "Imagens/video720p.mkv"

    findRectangle(source)

    preProcessingRoi()

    recognitionOCR()
