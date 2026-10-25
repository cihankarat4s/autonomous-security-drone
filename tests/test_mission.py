from security_drone.mavlink.mission import PatrolTimer, TriggerWatcher


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


def test_patrol_timer_not_started_never_expires():
    timer = PatrolTimer(duration_s=60)
    assert timer.expired(1000) is False


def test_patrol_timer_expires_after_duration():
    timer = PatrolTimer(duration_s=60)
    timer.start(100)
    assert timer.expired(159) is False
    assert timer.expired(160) is True


def test_patrol_timer_stop():
    timer = PatrolTimer(duration_s=60)
    timer.start(100)
    timer.stop()
    assert timer.running is False
    assert timer.expired(200) is False
