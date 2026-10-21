from pymavlink import mavutil

mavlink = mavutil.mavlink

# STATUSTEXT.text is a fixed 50-byte field; longer text would be cut silently.
MAX_TEXT_LEN = 50


def threat_text(person_count, lat_deg, lon_deg, when_utc):
    """E.g. 'Threat: 1 person at -35.36270,149.16524 12:34:56Z'.

    Time is UTC (the trailing Z) so it matches flight logs regardless of
    where the ground station is.
    """
    if lat_deg is None:
        where = "pos unknown"
    else:
        where = f"at {lat_deg:.5f},{lon_deg:.5f}"
    text = f"Threat: {person_count} person {where} {when_utc:%H:%M:%S}Z"
    return text[:MAX_TEXT_LEN]


def send_statustext(master, text, severity=mavlink.MAV_SEVERITY_ALERT):
    master.mav.statustext_send(severity, text[:MAX_TEXT_LEN].encode("ascii"))
