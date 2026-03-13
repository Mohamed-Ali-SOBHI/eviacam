import struct
import sys

import cv2
import mediapipe as mp
import numpy as np


RIGHT_EYE_INDEX = 33
LEFT_EYE_INDEX = 263
NOSE_INDEX = 1
RIGHT_MOUTH_INDEX = 61
LEFT_MOUTH_INDEX = 291


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def scaled_point(landmark, width, height):
    return (
        int(round(clamp(landmark.x, 0.0, 1.0) * (width - 1))),
        int(round(clamp(landmark.y, 0.0, 1.0) * (height - 1))),
    )


def build_face_result(face_landmarks, width, height):
    xs = [clamp(landmark.x, 0.0, 1.0) for landmark in face_landmarks.landmark]
    ys = [clamp(landmark.y, 0.0, 1.0) for landmark in face_landmarks.landmark]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    face_width = max_x - min_x
    face_height = max_y - min_y

    min_x = clamp(min_x - face_width * 0.14, 0.0, 1.0)
    max_x = clamp(max_x + face_width * 0.14, 0.0, 1.0)
    min_y = clamp(min_y - face_height * 0.22, 0.0, 1.0)
    max_y = clamp(max_y + face_height * 0.12, 0.0, 1.0)

    x = int(round(min_x * (width - 1)))
    y = int(round(min_y * (height - 1)))
    w = max(1, int(round((max_x - min_x) * width)))
    h = max(1, int(round((max_y - min_y) * height)))

    key_indices = [
        RIGHT_EYE_INDEX,
        LEFT_EYE_INDEX,
        NOSE_INDEX,
        RIGHT_MOUTH_INDEX,
        LEFT_MOUTH_INDEX,
    ]
    key_points = []
    for index in key_indices:
        px, py = scaled_point(face_landmarks.landmark[index], width, height)
        key_points.extend((px, py))

    return [x, y, w, h, 1.0] + key_points


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
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
            result = face_mesh.process(rgb_frame)

            if not result.multi_face_landmarks:
                stdout.write("NONE\n")
                stdout.flush()
                continue

            face_result = build_face_result(
                result.multi_face_landmarks[0], frame.shape[1], frame.shape[0]
            )
            stdout.write("OK " + " ".join(str(value) for value in face_result) + "\n")
            stdout.flush()
    finally:
        face_mesh.close()


if __name__ == "__main__":
    main()
