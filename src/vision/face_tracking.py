import cv2
from pymavlink import mavutil


# Load Haar Cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def process_frame(master, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    frame_center_x = frame.shape[1] // 2

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x = x + w // 2

        yaw = 0
        if face_center_x < frame_center_x - 50:
            yaw = -100  # Turn left
        elif face_center_x > frame_center_x + 50:
            yaw = 100   # Turn right

        # Maintain constant throttle and no roll or pitch
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
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
