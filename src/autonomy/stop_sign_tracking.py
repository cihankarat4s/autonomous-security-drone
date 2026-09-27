import cv2
import numpy as np
import math
from pymavlink import mavutil


def connect_to_pixhawk(connection_string, baudrate=115200):
    master = mavutil.mavlink_connection(connection_string, baud=baudrate)
    master.wait_heartbeat()
    print("Connected to Pixhawk.")
    return master

def arm_vehicle(master):
    master.arducopter_arm()
    master.motors_armed_wait()
    print("Drone is armed.")

def send_manual_control(master, x, y, z, r):
    master.mav.manual_control_send(
        master.target_system,
        x,  # Roll
        y,  # Pitch
        z,  # Throttle
        r,  # Yaw
        0   # Buttons
    )
    print(f"Sent manual control command: Roll={x}, Pitch={y}, Throttle={z}, Yaw={r}")

def is_convex(contour):
    return cv2.isContourConvex(contour)

def find_stop_sign(frame):
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_red)

    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    edges = cv2.Canny(gray_frame, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    largest_octagon_center = (0, 0)
    approx = None

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 8 and is_convex(approx):
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    x, y, w, h = cv2.boundingRect(approx)
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        largest_octagon_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if max_area > 0:
        cx, cy = largest_octagon_center
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
        distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        text = f"Coords: ({cx}, {cy}), Distance: {int(distance)}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y), (cx, cy), (255, 0, 0), 2)
        return cx, cy, center_x, center_y

    return None

def process_frame(master, frame):
    result = find_stop_sign(frame)
    if result:
        cx, cy, center_x, center_y = result

        yaw = 0
        if cx < center_x + 50:
            yaw = +100  # Turn left
        elif cx > center_x - 50:
            yaw = -100  # Turn right

        send_manual_control(master, 0, 0, 200, yaw)

def main():
    connection_string = 'COM9'
    master = connect_to_pixhawk(connection_string)

    if master is None:
        print("Failed to connect to Pixhawk.")
        return

    arm_vehicle(master)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(master, frame)
            cv2.imshow('STOP Sign Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


