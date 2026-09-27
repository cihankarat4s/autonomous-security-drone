from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
from pymavlink import mavutil

# Pixhawk bağlantısını kur
connection_string = 'tcp:127.0.0.1:5760'
print(f"Connecting to vehicle on: {connection_string}")

# Timeout ve heartbeat süresini artırın
vehicle = connect(connection_string, wait_ready=True, heartbeat_timeout=60)

def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def condition_yaw(vehicle, heading, relative=False):
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0,       # confirmation
        heading, # param 1, yaw in degrees
        0,       # param 2, yaw speed deg/s
        1,       # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    vehicle.send_mavlink(msg)

def move_forward(vehicle, distance):
    point1 = LocationGlobalRelative(vehicle.location.global_frame.lat,
                                    vehicle.location.global_frame.lon + (distance / 111139),
                                    vehicle.location.global_frame.alt)
    vehicle.simple_goto(point1)
    time.sleep(10)  # Bekleyerek hareketin tamamlanmasını sağlar

def main():
    arm_and_takeoff(2)  # 2 metreye kalkış
    condition_yaw(vehicle, 70)  # 70 derece sağa dönme
    time.sleep(2)  # Yaw komutunun tamamlanması için bekleme

    move_forward(vehicle, 10)  # 10 metre ileriye gitme
    time.sleep(10)  # İleri hareketin tamamlanması için bekleme

    print("Returning to Launch")
    vehicle.mode = VehicleMode("LAND")

    while vehicle.armed:
        print(" Waiting for disarming...")
        time.sleep(1)

    print("Close vehicle object")
    vehicle.close()

if __name__ == "__main__":
    main()
