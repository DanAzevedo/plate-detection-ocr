import cv2
import pytesseract
import matplotlib.pyplot as plt

img = cv2.imread('Imagens/teste_tesseract.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

boxes = pytesseract.image_to_data(img)

for b in boxes.splitlines():
    print(b)

for a, b in enumerate(boxes.splitlines()):
    if a != 0:
        b = b.split()
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x,y), (x+w, y+h), (50,50,255), 2)
            cv2.putText(img, b[11], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)

plt.imshow(img)
plt.show()