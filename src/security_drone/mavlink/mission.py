class TriggerWatcher:
    """Reports once when the vehicle reaches the configured mission item.

    The original code hard-coded seq == 1, but in a mission that starts with
    TAKEOFF, seq 1 is reached right after takeoff, above the launch point.
    The patrol waypoint's seq therefore comes from the config file.
    """

    def __init__(self, trigger_seq):
        self.trigger_seq = trigger_seq
        self.triggered = False

    def on_item_reached(self, seq):
        """Return True only the first time trigger_seq is reached."""
        if self.triggered or seq != self.trigger_seq:
            return False
        self.triggered = True
        return True

    def reset(self):
        """Arm the watcher again, e.g. after landing for the next mission."""
        self.triggered = False


class PatrolTimer:
    """Ends the patrol after a fixed time over the protected area.

    The original drone patrols until its mission time is up and the Pi then
    sends RTL. The time comes from the battery: ~10 min of flight on the
    6200 mAh pack, minus the flight out, the return and a reserve.
    """

    def __init__(self, duration_s):
        self.duration_s = duration_s
        self.started_at = None

    def start(self, now):
        self.started_at = now

    def stop(self):
        self.started_at = None

    @property
    def running(self):
        return self.started_at is not None

    def expired(self, now):
        """True once duration_s has passed since start(); False if not running."""
        return self.running and now - self.started_at >= self.duration_s
