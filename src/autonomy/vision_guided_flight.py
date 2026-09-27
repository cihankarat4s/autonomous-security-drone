import time
import cv2
import threading
from pymavlink import mavutil

# Drone ve kamera modÃ¼llerinin import edilmesi
from src.flight_control.drone_controller import connect_drone, arm_and_takeoff, land, send_movement_command_YAW, send_movement_command_XYA

# Initialize the last_command_time at the start of the script
last_command_time = time.time()  # Global variable for tracking command timing

def connect_to_pixhawk(connection_string, baudrate=115200):
    try:
        master = mavutil.mavlink_connection(connection_string, baud=baudrate)
        master.wait_heartbeat()
        print("Connection to Pixhawk established.")
        return master
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def arm_vehicle(master):
    try:
        print("Arming the drone...")
        master.arducopter_arm()
        master.motors_armed_wait()
        print("Drone armed.")
    except Exception as e:
        print(f"Arming error: {e}")

def send_motor_command(master, roll, pitch, throttle, yaw):
    try:
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            roll,  # Aileron (roll)
            pitch,  # Elevator (pitch)
            throttle,  # Throttle
            yaw,  # Rudder (yaw)
            0, 0, 0, 0)  # Remaining channels
        print(f"Motor command sent -> Roll: {roll}, Pitch: {pitch}, Throttle: {throttle}, Yaw: {yaw}")
    except Exception as e:
        print(f"Error sending motor command: {e}")

def process_frame(frame, master, frame_center_x, frame_center_y):
    global last_command_time
    current_time = time.time()
    if current_time - last_command_time < 5:
        return  # Skip processing if within 5 seconds of the last command

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 8:  # Checking for octagon
            x, y, w, h = cv2.boundingRect(approx)
            stop_sign_center_x, stop_sign_center_y = x + w // 2, y + h // 2
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.circle(frame, (stop_sign_center_x, stop_sign_center_y), 5, (0, 255, 0), -1)

            throttle = 1200  # Neutral throttle
            yaw = 1500  # Neutral yaw
            roll = 1500  # Neutral roll
            pitch = 1500  # Neutral pitch

            if stop_sign_center_x > frame_center_x + 50:  # Adjust yaw based on the stop sign position
                yaw = 1600
            elif stop_sign_center_x < frame_center_x - 50:
                yaw = 1400

            if stop_sign_center_y > frame_center_y + 50:  # Adjust pitch based on the stop sign position
                pitch = 1600
            elif stop_sign_center_y < frame_center_y - 50:
                pitch = 1400

            if w > 100:  # Adjust roll based on the width of the stop sign (indicating distance)
                roll = 1400
            elif w < 60:
                roll = 1600

            send_motor_command(master, roll, pitch, throttle, yaw)
            last_command_time = current_time
            break

def camera_thread(master, cam_ready_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    cam_ready_event.set()  # Signal that the camera is ready

    cap.set(3, 640)  # Set video width
    cap.set(4, 480)  # Set video height
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
            frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
            process_frame(frame, master, frame_center_x, frame_center_y)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    master = connect_to_pixhawk('COM9', 57600)
    if not master:
        return

    cam_ready_event = threading.Event()
    cam_thread = threading.Thread(target=camera_thread, args=(master, cam_ready_event))
    cam_thread.start()

    cam_ready_event.wait()  # Wait until the camera is ready
    arm_vehicle(master)  # Arm the vehicle after the camera is ready

    cam_thread.join()
    master.close()
    print("Connection closed")

if __name__ == "__main__":
    main()


