import enum
import logging

log = logging.getLogger("security_drone")


class AlarmEvent(enum.Enum):
    ACTIVATED = "activated"
    REMINDER = "reminder"
    CLEARED = "cleared"


class ThreatAlarm:
    """Decides when the alarm turns on, repeats and turns off.

    The original code slept 5 s while the buzzer sounded, which froze the camera
    and detection loop, and stopped after a fixed time even if the threat was
    still there. Here time is passed in, nothing sleeps, and the alarm stays
    on until nobody has been seen for clear_after_s.
    """

    def __init__(self, clear_after_s, notify_every_s):
        self.clear_after_s = clear_after_s
        self.notify_every_s = notify_every_s
        self.active = False
        self._last_seen = None
        self._last_notify = None

    def update(self, person_count, now):
        """Feed one detection result; returns an AlarmEvent or None."""
        if person_count > 0:
            self._last_seen = now
            if not self.active:
                self.active = True
                self._last_notify = now
                return AlarmEvent.ACTIVATED
            # Repeat the GCS message now and then, without spamming it.
            if now - self._last_notify >= self.notify_every_s:
                self._last_notify = now
                return AlarmEvent.REMINDER
            return None
        if self.active and now - self._last_seen >= self.clear_after_s:
            self.active = False
            return AlarmEvent.CLEARED
        return None


class LogAlarm:
    """Stands in for the GPIO buzzer in simulation."""

    def on(self):
        log.warning("ALARM ON (buzzer)")

    def off(self):
        log.info("Alarm off")
