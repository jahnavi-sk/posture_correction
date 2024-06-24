import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
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
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def preprocess_image(image, input_size):
    image = tf.image.resize_with_pad(image, input_size, input_size)
    image = tf.cast(image, dtype=tf.int32)
    return image

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
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
     edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

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
        rect = patches.Rectangle(
            (xmin, ymin), rec_width, rec_height,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_from_plot = np.asarray(rgba_buffer, dtype=np.uint8)

    fig_width, fig_height = fig.canvas.get_width_height()
    expected_total_size = fig_width * fig_height * 4  # rgba has 4 channels

    if image_from_plot.size != expected_total_size:
        print(f"Mismatch in sizes: Expected {expected_total_size}, got {image_from_plot.size}")
    else:
        image_from_plot = image_from_plot.reshape(fig_height, fig_width, 4)  # reshape to include the alpha channel
        image_from_plot = image_from_plot[..., :3]  # remove the alpha channel

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    
    plt.close(fig)
    return image_from_plot



def to_gif(images, fps):
    """Converts image sequence (4D numpy array) to gif."""
    imageio.mimsave('./animation.gif', images, fps=fps)
    return embed.embed_file('./animation.gif')

def progress(value, max=100):
    return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

model_name = "movenet_thunder"

if "tflite" in model_name:
    if "movenet_lightning_f16" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
        input_size = 192
    elif "movenet_thunder_f16" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
        input_size = 256
    elif "movenet_lightning_int8" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
        input_size = 192
    elif "movenet_thunder_int8" in model_name:
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    # Download the model
    import subprocess
    subprocess.run(["wget", "-q", "-O", "model.tflite", url], check=True)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    def movenet(input_image):
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

else:
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    def movenet(input_image):
        model = module.signatures['serving_default']

        # Ensure the input image is of the right shape: [1, input_size, input_size, 3]
        # input_image = tf.cast(input_image, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores


def smooth_keypoints(keypoints_with_scores, alpha=0.7):
    smoothed_keypoints = keypoints_with_scores.copy()
    for i in range(1, keypoints_with_scores.shape[1]):
        smoothed_keypoints[0, i, :, :2] = alpha * keypoints_with_scores[0, i, :, :2] + (1 - alpha) * keypoints_with_scores[0, i-1, :, :2]
    return smoothed_keypoints

# Example usage:
# Assuming you have an image loaded as `input_image`
input_image = tf.image.decode_image(tf.io.read_file('../media/5.png'))
# Remove the alpha channel if present
if input_image.shape[-1] == 4:
    input_image = input_image[..., :3]
preprocessed_image = preprocess_image(input_image, input_size)
keypoints_with_scores = movenet(input_image)
keypoints_with_scores = smooth_keypoints(keypoints_with_scores)
image_with_keypoints = draw_prediction_on_image(input_image.numpy(), keypoints_with_scores)
plt.imshow(image_with_keypoints)
plt.show()
