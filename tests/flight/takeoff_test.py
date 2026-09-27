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

def takeoff(master, altitude):
    # Set mode to GUIDED
    master.set_mode("GUIDED")
    master.motors_armed_wait()

    # Send takeoff command
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    print(f"Takeoff to {altitude} meters")

def main():
    connection_string = 'COM9'
    master = connect_to_pixhawk(connection_string)

    if master is None:
        print("Failed to connect to Pixhawk.")
        return

    arm_vehicle(master)
    takeoff(master, 2)  # Takeoff to 2 meters altitude

    # Keep the script running to maintain the connection
    try:
        while True:
            master.recv_match(blocking=True)
    except KeyboardInterrupt:
        print("Program interrupted by user.")

if __name__ == "__main__":
    main()

