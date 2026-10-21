from security_drone.mavlink.mission import TriggerWatcher


def test_takeoff_item_does_not_trigger():
    watcher = TriggerWatcher(trigger_seq=2)
    assert watcher.on_item_reached(1) is False


def test_trigger_fires_once():
    watcher = TriggerWatcher(trigger_seq=2)
    assert watcher.on_item_reached(2) is True
    assert watcher.on_item_reached(2) is False


def test_reset_allows_next_mission():
    watcher = TriggerWatcher(trigger_seq=2)
    watcher.on_item_reached(2)
    watcher.reset()
    assert watcher.on_item_reached(2) is True
