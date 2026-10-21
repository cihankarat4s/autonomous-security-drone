from security_drone.alert.alarm import AlarmEvent, ThreatAlarm


def make_alarm():
    return ThreatAlarm(clear_after_s=3, notify_every_s=10)


def test_first_detection_activates():
    alarm = make_alarm()
    assert alarm.update(1, now=0.0) is AlarmEvent.ACTIVATED
    assert alarm.active


def test_no_repeat_before_notify_period():
    alarm = make_alarm()
    alarm.update(1, now=0.0)
    assert alarm.update(1, now=5.0) is None


def test_reminder_after_notify_period():
    alarm = make_alarm()
    alarm.update(1, now=0.0)
    assert alarm.update(2, now=10.0) is AlarmEvent.REMINDER


def test_short_gap_keeps_alarm_on():
    alarm = make_alarm()
    alarm.update(1, now=0.0)
    assert alarm.update(0, now=2.0) is None
    assert alarm.active


def test_clears_after_timeout():
    alarm = make_alarm()
    alarm.update(1, now=0.0)
    assert alarm.update(0, now=3.5) is AlarmEvent.CLEARED
    assert not alarm.active


def test_nothing_happens_without_detections():
    alarm = make_alarm()
    assert alarm.update(0, now=100.0) is None
