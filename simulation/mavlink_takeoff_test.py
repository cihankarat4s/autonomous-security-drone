from pymavlink import mavutil

# Mission Planner simülasyonunda kullanılan UDP portuna bağlan
connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')

# MAVLink heartbeat sinyali bekle
connection.wait_heartbeat()
print("Heartbeat received")

# Örnek komut gönderimi (örneğin, kalkış komutu)
connection.mav.command_long_send(
    connection.target_system, 
    connection.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, # Confirmation
    0, 0, 0, 0, 0, 0, 10 # Parametreler (altitude 10 metre)
)
