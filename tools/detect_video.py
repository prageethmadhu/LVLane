from pathlib import Path
import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
import torch
import numpy as np
from tqdm import tqdm
from lanedet.core.lane import Lane
device = torch.device('cpu')  # Use 'cuda' if you have a GPU
import matplotlib.pyplot as plt

class VideoLaneDetection:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.DataParallel(self.net).to(device)
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, frame):
        ori_img = frame
        img = cv2.resize(ori_img[self.cfg.cut_height:, :, :], (800, 288))  # Resize to match model's expected input
        img = img.astype(np.float32).transpose(2, 0, 1)  # Convert to [channels, height, width]
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
        data = {'img': img.to(device), 'lanes': []}
        data['ori_img'] = ori_img
        return data


    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)  # Ensure input is correctly preprocessed
            data = self.net.module.get_lanes(data)
        return data




    def inference(self, data):
        """Run inference on the preprocessed frame."""
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def visualize(self, data, out_file=None):
        # Draw the lanes on the image
        for lane in data['lanes']:
            # Ensure the lane is iterable and contains coordinate pairs
            if isinstance(lane, np.ndarray) and len(lane.shape) == 2 and lane.shape[1] == 2:
                for x, y in lane:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    cv2.circle(data['ori_img'], (x, y), 4, (0, 255, 0), 2)
            else:
                print(f"Unexpected lane format: {lane}")

        # Save or display the output
        if out_file:
            cv2.imwrite(out_file, data['ori_img'])

    def process_video(self, input_path, output_path=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video {input_path}")

        video_writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            # Ensure data is always initialized
            data = None

            try:
                data = self.preprocess(frame)
                data['lanes'] = self.inference(data)['lane_output']
                self.visualize(data, out_file=None)

                # Write to output file if writer is initialized
                if video_writer:
                    video_writer.write(data['ori_img'])

                # Show the video frame with detections
                cv2.imshow("Lane Detection", data['ori_img'])

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue  # Skip to the next frame

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--video', help='Path to the input video file')
    parser.add_argument('--output', help='Path to save the output video file')
    parser.add_argument('--load_from', type=str, default='best.pth', help='Path to the pretrained model')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    video_detector = VideoLaneDetection(cfg)
    video_detector.process_video(args.video, args.output)
