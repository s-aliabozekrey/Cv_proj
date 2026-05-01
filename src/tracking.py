import math


class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def register(self, bbox, class_id):
        centroid = self._get_centroid(bbox)
        self.objects[self.next_object_id] = {
            "bbox": bbox,
            "centroid": centroid,
            "class_id": class_id
        }
        self.next_object_id += 1

    def update(self, detections):
        """
        detections format:
        [
            {"bbox": (x1, y1, x2, y2), "class_id": 0, "confidence": 0.95},
            ...
        ]
        """
        if len(self.objects) == 0:
            for det in detections:
                self.register(det["bbox"], det["class_id"])
            return self.objects

        updated_objects = {}
        used_detection_indices = set()

        object_items = list(self.objects.items())

        for object_id, obj_data in object_items:
            old_centroid = obj_data["centroid"]
            old_class_id = obj_data["class_id"]

            best_match_index = -1
            best_distance = float("inf")

            for i, det in enumerate(detections):
                if i in used_detection_indices:
                    continue

                if det["class_id"] != old_class_id:
                    continue

                new_centroid = self._get_centroid(det["bbox"])
                distance = self._euclidean_distance(old_centroid, new_centroid)

                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match_index = i

            if best_match_index != -1:
                det = detections[best_match_index]
                used_detection_indices.add(best_match_index)

                updated_objects[object_id] = {
                    "bbox": det["bbox"],
                    "centroid": self._get_centroid(det["bbox"]),
                    "class_id": det["class_id"]
                }

        for i, det in enumerate(detections):
            if i not in used_detection_indices:
                updated_objects[self.next_object_id] = {
                    "bbox": det["bbox"],
                    "centroid": self._get_centroid(det["bbox"]),
                    "class_id": det["class_id"]
                }
                self.next_object_id += 1

        self.objects = updated_objects
        return self.objects