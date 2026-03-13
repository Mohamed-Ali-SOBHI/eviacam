import struct
import sys

import cv2
import mediapipe as mp
import numpy as np


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def scaled_point(landmark, width, height):
    return (
        int(round(clamp(landmark.x, 0.0, 1.0) * (width - 1))),
        int(round(clamp(landmark.y, 0.0, 1.0) * (height - 1))),
    )


def build_face_result(detection, detector, width, height):
    bbox = detection.location_data.relative_bounding_box

    min_x = clamp(bbox.xmin, 0.0, 1.0)
    min_y = clamp(bbox.ymin, 0.0, 1.0)
    box_width = clamp(bbox.width, 0.0, 1.0 - min_x)
    box_height = clamp(bbox.height, 0.0, 1.0 - min_y)

    x = int(round(min_x * (width - 1)))
    y = int(round(min_y * (height - 1)))
    w = max(1, int(round(box_width * width)))
    h = max(1, int(round(box_height * height)))

    score = detection.score[0] if detection.score else 0.0

    key_points = []
    key_names = [
        detector.FaceKeyPoint.RIGHT_EYE,
        detector.FaceKeyPoint.LEFT_EYE,
        detector.FaceKeyPoint.NOSE_TIP,
        detector.FaceKeyPoint.MOUTH_CENTER,
        detector.FaceKeyPoint.MOUTH_CENTER,
    ]
    for key_name in key_names:
        point = detector.get_key_point(detection, key_name)
        px, py = scaled_point(point, width, height)
        key_points.extend((px, py))

    return [x, y, w, h, score] + key_points


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.1,
    )

    stdout = sys.stdout
    stdin = sys.stdin.buffer

    stdout.write("READY\n")
    stdout.flush()

    try:
        while True:
            header = read_exact(stdin, 4)
            if header is None:
                break

            (payload_size,) = struct.unpack("<I", header)
            payload = read_exact(stdin, payload_size)
            if payload is None:
                break

            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                stdout.write("NONE\n")
                stdout.flush()
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_detection.process(rgb_frame)

            if not result.detections:
                stdout.write("NONE\n")
                stdout.flush()
                continue

            face_result = build_face_result(
                result.detections[0],
                mp.solutions.face_detection,
                frame.shape[1],
                frame.shape[0],
            )
            stdout.write("OK " + " ".join(str(value) for value in face_result) + "\n")
            stdout.flush()
    finally:
        face_detection.close()


if __name__ == "__main__":
    main()
