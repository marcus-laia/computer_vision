import cv2 as cv
from pyzbar.pyzbar import decode

img = cv.imread('resources/qr-code.jpg')

for barcode in decode(img):
    print(barcode.data)
    data = barcode.data.decode('utf-8')
    print(data)