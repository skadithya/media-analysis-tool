def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.
    """

    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = (
        (box1["x2"] - box1["x1"])
        * (box1["y2"] - box1["y1"])
    )

    area2 = (
        (box2["x2"] - box2["x1"])
        * (box2["y2"] - box2["y1"])
    )

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def track_objects(frame_detections, iou_threshold=0.3):
    """
    Track objects across sampled video frames.

    frame_detections format:

    [
        {
            "timestamp": 0,
            "detections": [...]
        },
        {
            "timestamp": 5,
            "detections": [...]
        }
    ]
    """

    tracks = []
    next_track_id = 1

    for frame in frame_detections:

        timestamp = frame["timestamp"]
        detections = frame["detections"]

        for detection in detections:

            object_name = detection["object"]
            current_box = detection["bounding_box"]

            best_track = None
            best_iou = 0.0

            # Compare with existing tracks
            for track in tracks:

                # Only compare the same object class
                if track["object"] != object_name:
                    continue

                # Track must have appeared in the previous frame
                if track["last_timestamp"] != timestamp - 5:
                    continue

                iou = calculate_iou(
                    track["last_box"],
                    current_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            # Existing object found
            if best_track and best_iou >= iou_threshold:

                best_track["last_timestamp"] = timestamp
                best_track["last_box"] = current_box

                best_track["detections"].append({
                    "timestamp": timestamp,
                    "confidence": detection["confidence"],
                    "bounding_box": current_box
                })

            # New object
            else:

                new_track = {
                    "track_id": next_track_id,
                    "object": object_name,
                    "first_timestamp": timestamp,
                    "last_timestamp": timestamp,
                    "last_box": current_box,
                    "detections": [
                        {
                            "timestamp": timestamp,
                            "confidence": detection["confidence"],
                            "bounding_box": current_box
                        }
                    ]
                }

                tracks.append(new_track)
                next_track_id += 1

    return tracks