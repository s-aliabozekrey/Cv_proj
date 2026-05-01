from tracking import CentroidTracker

# Persistent tracker and initial pin set across frames
tracker = CentroidTracker(max_distance=60)
TOTAL_PINS = 10

# Movement-based knockdown tracking state
BASELINE_FRAMES = 10
MOVE_THRESHOLD_PX = 30
baseline_locked = False
frame_index = 0
baseline_positions = []
best_baseline_count = 0


def _distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def count_fallen_pins(detections):
    """
    Update tracker with current detections and estimate knocked-down pins.

    Returns: knocked_down
    """
    global baseline_locked, frame_index, baseline_positions, best_baseline_count

    objects = tracker.update(detections)

    # Gather current pin objects (class_id == 0).
    current_pins = [
        data["centroid"]
        for data in objects.values()
        if data["class_id"] == 0
    ]

    # Build a stable baseline snapshot within the first few frames.
    if not baseline_locked:
        if len(current_pins) > best_baseline_count:
            best_baseline_count = len(current_pins)
            baseline_positions = list(current_pins)

        if frame_index >= BASELINE_FRAMES and len(baseline_positions) > 0:
            baseline_locked = True

    frame_index += 1

    if not baseline_locked:
        return 0

    # Match baseline positions to current positions by nearest distance.
    remaining_current = list(current_pins)
    standing_count = 0
    for base_pos in baseline_positions:
        if not remaining_current:
            break

        nearest_index = -1
        nearest_distance = float("inf")
        for idx, cur_pos in enumerate(remaining_current):
            dist = _distance(base_pos, cur_pos)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_index = idx

        if nearest_index != -1 and nearest_distance <= MOVE_THRESHOLD_PX:
            standing_count += 1
            remaining_current.pop(nearest_index)

    pin_baseline = max(TOTAL_PINS, len(baseline_positions))
    knocked_down = pin_baseline - standing_count
    return min(pin_baseline, max(0, knocked_down))