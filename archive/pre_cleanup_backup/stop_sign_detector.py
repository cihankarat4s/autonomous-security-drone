import cv2
import numpy as np
import math

# Kamerayı aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılmadı")
    exit()

def find_stop_sign(frame):
    # Görüntüyü yatay olarak ters çevir
    frame = cv2.flip(frame, 1)

    # Ekranın ortasını hesapla
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    # Ekranın ortasına kırmızı bir nokta koy
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # BGR'den HSV'ye dönüştür
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için genişletilmiş HSV aralığı
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Maskeleri oluştur
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Maskeyi kullanarak sadece kırmızı nesneleri çıkart
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_red)

    # Grayscale dönüştür ve bulanıklaştır
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Kenarları algıla
    edges = cv2.Canny(gray_frame, 50, 150)

    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    largest_octagon_center = (0, 0)

    for contour in contours:
        # Konturun kapalı bir şekil olup olmadığını kontrol et
        if cv2.contourArea(contour) > 1000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Sekizgen olup olmadığını kontrol et
            if len(approx) == 8:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    x, y, w, h = cv2.boundingRect(approx)
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        largest_octagon_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if max_area > 0:
        cx, cy = largest_octagon_center
        # Sekizgenin merkezini ve ekranın ortasından olan uzaklığı çiz
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
        distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        text = f"Coords: ({cx}, {cy}), Distance: {int(distance)}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y), (cx, cy), (255, 0, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = find_stop_sign(frame)
    cv2.imshow('STOP Sign Tracker', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
