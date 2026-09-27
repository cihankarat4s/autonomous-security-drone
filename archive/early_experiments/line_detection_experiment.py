import cv2
import numpy as np
import time

# Motor pinlerini tanımlayın
left_motor_forward = 17
left_motor_backward = 18
right_motor_forward = 22
right_motor_backward = 23



# Kameradan görüntü al
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Siyah çizgiyi tespit etmek için eşikleme
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # ROI (Region of Interest) belirleme
    height, width = thresh.shape
    mask = np.zeros_like(thresh)
    polygon = np.array([[
        (0, height * 3 / 4),
        (width, height * 3 / 4),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(thresh, mask)

    # Çizgileri Hough Line Transformation ile tespit etme
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=50)
    
    # Orta noktayı belirleme
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            
        x_coords = [line[0][0] for line in lines] + [line[0][2] for line in lines]
        x_center = sum(x_coords) / len(x_coords)
        frame_center = width / 2

    # Görüntüyü göster
    cv2.imshow('frame', frame)
    
    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()

