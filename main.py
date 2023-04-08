import cv2
import numpy as np
import argparse
import math
from scipy.optimize import linear_sum_assignment
import os

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="video.mp4")
args = parser.parse_args()


def load_yolo():
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    return net, output_layers


def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap


def start_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calculate_cost_matrix(positions1, positions2):
    cost_matrix = np.zeros((len(positions1), len(positions2)))
    for i, pos1 in enumerate(positions1):
        for j, pos2 in enumerate(positions2):
            cost_matrix[i, j] = euclidean_distance(pos1, pos2)
    return cost_matrix


def detect_crash(net, output_layers, cap):
    vehicle_positions = []
    prev_vehicle_positions = []
    frame_count = 0
    crash_count = 0

    # Create the crash_frames folder if it doesn't exist
    if not os.path.exists("crash_frames"):
        os.makedirs("crash_frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        conf_threshold = 0.25   ### Adjust as per your requirement
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id == 2:
                    center_x, center_y, w, h = (detection[0:4] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.2)

        vehicle_positions = [
            (x + w // 2, y + h // 2) for i in np.array(indices).flatten() for x, y, w, h in [boxes[i]]]

        if frame_count > 0 and vehicle_positions and prev_vehicle_positions:
            cost_matrix = calculate_cost_matrix(
                prev_vehicle_positions, vehicle_positions)
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                prev_pos = prev_vehicle_positions[row]
                current_pos = vehicle_positions[col]
                distance = euclidean_distance(prev_pos, current_pos)

                if distance < 10:   ### Adjust as per your requirement
                    velocity = cost_matrix[row, col]

                    if velocity > 9:   ### Adjust as per your requirement
                        print("Crash detected")
                        cv2.putText(frame, "Crash Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                        # Save the frame as an image file
                        crash_count += 1
                        crash_frame_path = f"crash_frames/crash_{crash_count:04d}.png"
                        cv2.imwrite(crash_frame_path, frame)
                        print(f"Crash frame saved to {crash_frame_path}")
                        break

        prev_vehicle_positions = vehicle_positions
        frame_count += 1

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam = args.webcam
    video_path = args.video_path
    net, output_layers = load_yolo()

    if webcam:
        print('---- Starting Web Cam crash detection ----')
        cap = start_webcam()
    else:
        print('---- Starting Video crash detection ----')
        cap = start_video(video_path)

    detect_crash(net, output_layers, cap)
    cv2.destroyAllWindows()
