import datetime

from security_drone.alert.statustext import MAX_TEXT_LEN, threat_text

WHEN = datetime.datetime(2026, 10, 21, 12, 34, 56, tzinfo=datetime.timezone.utc)


def test_threat_text_contents():
    text = threat_text(1, -35.3627, 149.16524, WHEN)
    assert text == "Threat: 1 person at -35.36270,149.16524 12:34:56Z"


def test_threat_text_fits_statustext():
    text = threat_text(12345, -35.3627, 149.16524, WHEN)
    assert len(text) <= MAX_TEXT_LEN


def test_threat_text_without_position():
    assert "pos unknown" in threat_text(1, None, None, WHEN)
