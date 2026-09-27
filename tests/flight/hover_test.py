import time
from dronekit import connect, VehicleMode, LocationGlobalRelative

def connect_to_pixhawk(connection_string, baudrate=57600):
    try:
        vehicle = connect(connection_string, wait_ready=True, baud=baudrate)
        print("Connected to Pixhawk")
        return vehicle
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def arm_and_takeoff(vehicle, target_altitude):
    """
    Arms vehicle and fly to a target altitude with automatic stabilization.
    """
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def land(vehicle):
    print("Setting LAND mode...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(" Waiting for landing...")
        time.sleep(1)
    print("Landed")

def main():
    connection_string = 'COM9'  # Pixhawk bağlantı string'i. Linux'ta /dev/ttyACM0 gibi olur
    vehicle = connect_to_pixhawk(connection_string)

    if vehicle is None:
        print("Failed to connect to vehicle.")
        return

    try:
        target_altitude = 1  # 1 metreye yükselecek
        arm_and_takeoff(vehicle, target_altitude)

        # 10 saniye boyunca sabit konumda kal
        print("Hovering for 10 seconds")
        time.sleep(10)

        land(vehicle)
    finally:
        vehicle.close()
        print("Connection closed")

if __name__ == "__main__":
    main()

