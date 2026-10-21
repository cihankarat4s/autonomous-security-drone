class TriggerWatcher:
    """Reports once when the vehicle reaches the configured mission item.

    The thesis code hard-coded seq == 1, but in a mission that starts with
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
