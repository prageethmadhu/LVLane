import torch
from lanedet.ops import nms_impl

boxes = torch.tensor([[50, 50, 100, 100], [60, 60, 110, 110], [10, 10, 20, 20]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.75, 0.6], dtype=torch.float32)
selected_indices = nms_impl.nms_forward(boxes, scores, 0.5, 2)
print("Selected indices:", selected_indices)
