import cv2

bardet = cv2.barcode_BarcodeDetector()
img = cv2.imread("6.png")
ok, decoded_info, decoded_type, corners = bardet.detectAndDecode(img)