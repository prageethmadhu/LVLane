#undef HAVE_SNPRINTF
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <algorithm>

// Helper function to calculate IoU
float iou(const at::Tensor &box1, const at::Tensor &box2) {
    float x1 = std::max(box1[0].item<float>(), box2[0].item<float>());
    float y1 = std::max(box1[1].item<float>(), box2[1].item<float>());
    float x2 = std::min(box1[2].item<float>(), box2[2].item<float>());
    float y2 = std::min(box1[3].item<float>(), box2[3].item<float>());

    float intersection = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
    float box1_area = (box1[2].item<float>() - box1[0].item<float>() + 1) *
                      (box1[3].item<float>() - box1[1].item<float>() + 1);
    float box2_area = (box2[2].item<float>() - box2[0].item<float>() + 1) *
                      (box2[3].item<float>() - box2[1].item<float>() + 1);

    return intersection / (box1_area + box2_area - intersection);
}

// CPU-based NMS implementation
std::vector<at::Tensor> nms_cpu_forward(
        at::Tensor boxes,
        at::Tensor scores,
        float nms_overlap_thresh,
        unsigned long top_k) {
    boxes = boxes.contiguous();
    scores = scores.contiguous();

    auto indices = std::get<1>(scores.sort(0, true)); // Sort scores in descending order
    at::Tensor selected_indices = at::empty({0}, at::kLong);
    std::vector<bool> suppressed(boxes.size(0), false);

    for (int i = 0; i < indices.size(0); ++i) {
        if (suppressed[i]) continue;

        int64_t idx = indices[i].item<int64_t>();
        selected_indices = at::cat({selected_indices, at::tensor({idx}, at::kLong)});

        if (selected_indices.size(0) >= top_k) break;

        for (int j = i + 1; j < indices.size(0); ++j) {
            if (suppressed[j]) continue;

            float overlap = iou(boxes[idx], boxes[indices[j].item<int64_t>()]);
            if (overlap > nms_overlap_thresh) {
                suppressed[j] = true;
            }
        }
    }

    return {selected_indices};
}

// Wrapper function
std::vector<at::Tensor> nms_forward(
        at::Tensor boxes,
        at::Tensor scores,
        float thresh,
        unsigned long top_k) {
    return nms_cpu_forward(boxes, scores, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_forward", &nms_forward, "NMS (CPU)");
}
