from security_drone.detection.detector import Detection, merge_overlapping


def test_nested_boxes_on_one_person_become_one():
    body = Detection(100, 100, 110, 120, 0.8)
    with_shadow = Detection(90, 98, 112, 122, 0.6)
    inner = Detection(102, 104, 108, 116, 0.5)
    assert merge_overlapping([with_shadow, inner, body]) == [body]


def test_people_side_by_side_are_kept():
    left = Detection(100, 100, 110, 120, 0.7)
    right = Detection(108, 100, 118, 120, 0.6)   # overlaps 2 px of 10
    far = Detection(300, 300, 310, 320, 0.5)
    assert merge_overlapping([far, right, left]) == [left, right, far]


def test_most_confident_box_wins():
    weak = Detection(100, 100, 110, 120, 0.3)
    strong = Detection(99, 99, 111, 121, 0.9)
    assert merge_overlapping([weak, strong]) == [strong]


def test_no_detections():
    assert merge_overlapping([]) == []
