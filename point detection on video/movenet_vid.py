import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import display, HTML

# Helper functions and constants
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    # 'left_shoulder': 5,
    # 'right_shoulder': 6,
    # 'left_elbow': 7,
    # 'right_elbow': 8,
    # 'left_wrist': 9,
    # 'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    # (0, 5): 'm',
    # (0, 6): 'c',
    # (5, 7): 'm',
    # (7, 9): 'm',
    # (6, 8): 'c',
    # (8, 10): 'c',
    # (5, 6): 'y',
    # (5, 11): 'm',
    # (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)

    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(image, keypoints_with_scores, crop_region=None, close_figure=False, output_image_height=None):
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle((xmin, ymin), rec_width, rec_height, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()

    fig_width, fig_height = fig.canvas.get_width_height()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_from_plot = np.asarray(rgba_buffer)[:,:,:3]  # Slice off the alpha channel

    expected_total_size = fig_width * fig_height * 3
    if image_from_plot.size != expected_total_size:
        print(f"Mismatch in sizes: Expected {expected_total_size}, got {image_from_plot.size}")
    else:
        image_from_plot = image_from_plot.reshape(fig_height, fig_width, 3)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(image_from_plot, dsize=(output_image_width, output_image_height), interpolation=cv2.INTER_CUBIC)
    plt.close(fig)
    return image_from_plot

def movenet(input_image):
    model = module.signatures['serving_default']
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

# Load the MoveNet model
model_name = "movenet_thunder"
if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)

# Process video
video_path = './raise_10.mp4'
output_video_path = './raise_10_yess.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    input_image = tf.convert_to_tensor(frame, dtype=tf.uint8)
    if input_image.shape[-1] == 4:
        input_image = input_image[..., :3]
    keypoints_with_scores = movenet(input_image)
    image_with_keypoints = draw_prediction_on_image(input_image.numpy(), keypoints_with_scores)
    if out is None:
        height, width, _ = image_with_keypoints.shape
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    out.write(cv2.cvtColor(image_with_keypoints, cv2.COLOR_RGB2BGR))
    print(f"Processed frame {i + 1}/{frame_count}")

cap.release()
out.release()

# Display the resulting video
display(HTML(f'<video controls><source src="{output_video_path}" type="video/mp4"></video>'))
