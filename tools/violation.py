import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import glob
import torch
import numpy as np
import argparse
import logging
import random
from tqdm import tqdm
from pathlib import Path
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from lanedet.core.lane import Lane
from ultralytics import YOLO  # For vehicle detection
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set up logging to display debug messages.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Set device – change to 'cuda' if a GPU is available.
device = torch.device('cpu')

def compute_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class VideoLaneDetection:
    def __init__(self, cfg, yolo_model_path='yolov8n.pt'):
        """Initialize lane detection model, YOLO vehicle detector, and tracker."""
        self.cfg = cfg
        self.processes = Process(cfg.infer_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids=range(1)).to(device)
        total_params = sum(param.numel() for param in self.net.parameters())
        logging.info(f"total number of params: {total_params}")
        self.net.eval()
        load_network(self.net, self.cfg.load_from)
        self.yolo_model = YOLO(yolo_model_path)
        # Tracker variables.
        # Each track is stored as: { track_id: { "box": (x1,y1,x2,y2), "lost": count } }
        self.tracks = {}
        self.next_vehicle_id = 0
        self.iou_threshold = 0.3  # Minimum IoU to match a track.
        self.lost_threshold = 3   # Frames to wait before removing a track.

    def preprocess(self, frame):
        """Prepare image for lane detection model."""
        ori_img = frame.copy()
        img = cv2.resize(ori_img[self.cfg.cut_height:, :, :], (800, 288))
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        data = {'img': img.to(device), 'lanes': []}
        data['ori_img'] = ori_img
        return data

    def detect_vehicles(self, frame):
        """Run YOLO inference to detect vehicles in a frame."""
        results = self.yolo_model(frame)[0]
        vehicle_boxes = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) in [2, 3, 5, 7]:
                vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return vehicle_boxes

    def update_tracks(self, new_boxes):
        """
        Update tracks using IoU matching.
        Each new box is compared against existing tracks.
        If a box has IoU > threshold with an existing track, update that track;
        otherwise, create a new track.
        Also, increment lost counter for unmatched tracks and remove if lost too many frames.
        Returns a list of tuples: (vehicle_id, box)
        """
        assignments = []
        updated_tracks = {}

        # First, match new boxes to existing tracks.
        for box in new_boxes:
            best_iou = 0.0
            best_id = None
            for track_id, track in self.tracks.items():
                iou = compute_iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_iou > self.iou_threshold and best_id is not None:
                assignments.append((best_id, box))
                updated_tracks[best_id] = {"box": box, "lost": 0}
            else:
                # Create new track.
                new_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                assignments.append((new_id, box))
                updated_tracks[new_id] = {"box": box, "lost": 0}

        # For tracks not updated, increase lost count.
        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                track["lost"] += 1
                if track["lost"] < self.lost_threshold:
                    updated_tracks[track_id] = track
        self.tracks = updated_tracks
        return assignments

    def inference(self, data):
        """Run lane detection inference."""
        with torch.no_grad():
            outputs = self.net(data)
            lanes_dict = self.net.module.get_lanes(outputs)
            if self.cfg.classification:
                lane_detection, lane_indx = list(lanes_dict.values())
                lane_classes = self.get_lane_class(outputs, lane_indx)
                return lane_detection[0], lane_classes
            else:
                return lanes_dict['lane_output']

    def get_lane_class(self, predictions, lane_indx):
        """Compute lane classes from the network predictions."""
        score = F.softmax(predictions['category'], dim=1)
        y_pred = score.argmax(dim=1).squeeze()
        return y_pred[lane_indx].detach().cpu().numpy()

    def get_lane_color(self, lane_type):
        """Return a BGR color based on the lane type."""
        lane_color_map = {
            "solid": (0, 255, 255),   # Yellow
            "dashed": (255, 0, 255),  # Magenta
            "double": (255, 255, 0),  # Cyan
            "default": (0, 255, 0)    # Green
        }
        color = lane_color_map.get(lane_type, lane_color_map["default"])
        logging.debug(f"Lane type '{lane_type}' maps to color {color}")
        return color

    def get_lane_type_from_class(self, lane_class):
        """Map numeric lane class to a string lane type."""
        mapping = {0: "solid", 1: "dashed", 2: "double"}
        return mapping.get(lane_class, "solid")

    def visualize(self, data, tracked_vehicle_boxes, lane_classes=None, out_file=None):
        """
        Visualize detected lanes and tracked vehicles.
        
        - Draw each lane with its assigned color.
        - Select the mid-line by sorting lanes by average normalized x and taking the second from the left.
        - Only if the mid-line is "dashed" (magenta) will its points be used for violation detection.
        - Draw each tracked vehicle box with its vehicle ID.
        """
        img = data['ori_img'].copy()
        all_lanes = data.get('lanes', [])
        if not isinstance(all_lanes, list):
            all_lanes = [all_lanes]
        
        # Compute average normalized x for each lane.
        lane_avg_x = []
        for i, lane in enumerate(all_lanes):
            if isinstance(lane, Lane) and lane.points is not None and len(lane.points) > 0:
                avg_x = np.mean(lane.points[:, 0])
                lane_avg_x.append((i, avg_x))
            else:
                lane_avg_x.append((i, float('inf')))
        lane_avg_x.sort(key=lambda item: item[1])
        logging.debug(f"Sorted lanes by average x: {lane_avg_x}")
        if len(lane_avg_x) >= 2:
            mid_line_idx = lane_avg_x[1][0]
        elif len(lane_avg_x) > 0:
            mid_line_idx = lane_avg_x[0][0]
        else:
            mid_line_idx = None
        logging.debug(f"Selected mid-line index (second from left): {mid_line_idx}")
        
        mid_line_type = None
        if mid_line_idx is not None:
            if lane_classes is not None and mid_line_idx < len(lane_classes):
                mid_line_type = self.get_lane_type_from_class(lane_classes[mid_line_idx])
            else:
                mid_line_type = getattr(all_lanes[mid_line_idx], 'lane_type', "solid")
            logging.debug(f"Mid-line lane type: {mid_line_type}")
        
        violation_lane_points = []
        for idx, lane in enumerate(all_lanes):
            if isinstance(lane, Lane):
                if lane_classes is not None and idx < len(lane_classes):
                    lane_type = self.get_lane_type_from_class(lane_classes[idx])
                    logging.debug(f"Lane {idx}: using classified type '{lane_type}'")
                else:
                    lane_type = getattr(lane, 'lane_type', "solid")
                    logging.debug(f"Lane {idx}: using attribute type '{lane_type}'")
                color = self.get_lane_color(lane_type)
                pts = lane.points
                if pts is not None:
                    for x, y in pts:
                        if x > 0 and y > 0:
                            px = int(x * img.shape[1])
                            py = int(y * img.shape[0])
                            if idx == mid_line_idx and mid_line_type == "dashed":
                                violation_lane_points.append((px, py))
                            cv2.circle(img, (px, py), 4, color, 2)
                else:
                    logging.warning("Lane.points is None.")
            elif isinstance(lane, list):
                for j, sub_lane in enumerate(lane):
                    if isinstance(sub_lane, Lane):
                        if lane_classes is not None and idx < len(lane_classes):
                            lane_type = self.get_lane_type_from_class(lane_classes[idx])
                            logging.debug(f"Sub-lane {idx}-{j}: using classified type '{lane_type}'")
                        else:
                            lane_type = getattr(sub_lane, 'lane_type', "solid")
                            logging.debug(f"Sub-lane {idx}-{j}: using attribute type '{lane_type}'")
                        color = self.get_lane_color(lane_type)
                        pts = sub_lane.points
                        if pts is not None:
                            for x, y in pts:
                                if x > 0 and y > 0:
                                    px = int(x * img.shape[1])
                                    py = int(y * img.shape[0])
                                    if idx == mid_line_idx and mid_line_type == "dashed":
                                        violation_lane_points.append((px, py))
                                    cv2.circle(img, (px, py), 4, color, 2)
                        else:
                            logging.warning("Sub-lane.points is None.")
            else:
                logging.warning("Unexpected lane format encountered.")
        
        # Process each tracked vehicle box (with persistent vehicle IDs).
        for vehicle_id, (x1, y1, x2, y2) in tracked_vehicle_boxes:
            box_width = x2 - x1
            box_height = y2 - y1
            inner_width = int(box_width * 0.8)
            inner_height = int(box_height * 0.3)
            inner_x1 = x1 + (box_width - inner_width) // 2
            inner_y2 = y2
            inner_y1 = inner_y2 - inner_height
            inner_x2 = inner_x1 + inner_width

            violation = any(inner_x1 <= lx <= inner_x2 and inner_y1 <= ly <= inner_y2 
                            for (lx, ly) in violation_lane_points)
            if violation:
                outer_color = (0, 0, 255)
                label = "Violation"
                logging.debug(f"Vehicle {vehicle_id} box {(x1, y1, x2, y2)} flagged as violation.")
            else:
                outer_color = (0, 255, 0)
                label = "Vehicle"
                logging.debug(f"Vehicle {vehicle_id} box {(x1, y1, x2, y2)} is clear.")

            cv2.rectangle(img, (x1, y1), (x2, y2), outer_color, 2)
            cv2.putText(img, f"{label} {vehicle_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, outer_color, 2)
            cv2.rectangle(img, (inner_x1, inner_y1), (inner_x2, inner_y2), (255, 0, 0), 2)

        if out_file:
            cv2.imwrite(out_file, img)
            logging.info(f"Saved visualization to {out_file}")
        data['ori_img'] = img

    def process_video(self, input_path, output_path=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video {input_path}")
        video_writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or failed to read frame.")
                break
            try:
                data = self.preprocess(frame)
                if self.cfg.classification:
                    lanes, lane_classes = self.inference(data)
                    data['lanes'] = lanes
                else:
                    data['lanes'] = self.inference(data)
                    lane_classes = None
                new_boxes = self.detect_vehicles(frame)
                tracked_vehicle_boxes = self.update_tracks(new_boxes)
                self.visualize(data, tracked_vehicle_boxes, lane_classes=lane_classes, out_file=None)
                if video_writer:
                    video_writer.write(data['ori_img'])
                cv2.imshow("Lane & Vehicle Violation Detection", data['ori_img'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                continue
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video file')
    parser.add_argument('--load_from', type=str, default='best.pth', help='Path to the pretrained lane model')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    video_detector = VideoLaneDetection(cfg, args.yolo_model)
    video_detector.process_video(args.video, args.output)
