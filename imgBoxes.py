import cv2
import pytesseract
import matplotlib.pyplot as plt

img = cv2.imread('Imagens/teste_tesseract.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#plt.imshow(img)
#plt.show()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

boxes = pytesseract.image_to_boxes(img)
print(boxes)

hImg, wImg, _ = img.shape
print(hImg)
print(wImg)

for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50,50,255), 2)
    cv2.putText(img, b[0], (x, hImg - y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)

plt.imshow(img)
plt.show()