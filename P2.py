import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def stack_edges(image):
    c_mask, g_mask, _ = find_edges(image)

    mask = np.zeros_like(g_mask)
    mask[(g_mask == 1) | (c_mask == 1)] = 1
    return mask


def single_edges(image):
    c_mask, g_mask, s = find_edges(image)

    return np.dstack((np.zeros_like(s), g_mask, c_mask))


def find_edges(image):
    # Convert to HLS space
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    # Separate and get the S channel
    s = hls[:, :, 2]
    # Get all gradients
    grad_x = _gradient_absolute_value_mask(s, axis='x', threshold=(20, 100))
    grad_y = _gradient_absolute_value_mask(s, axis='y', threshold=(20, 100))
    magnit = _gradient_magnitude_mask(s, threshold=(20, 100))
    direct = _gradient_direction_mask(s, threshold=(0.7, 1.3))
    g_mask = np.zeros_like(s)
    g_mask[((grad_x == 1) & (grad_y == 1)) | ((magnit == 1) & (direct == 1))] = 1

    c_mask = _mask_between_thresholds(s, threshold=(170, 255))
    return c_mask, g_mask, s


def _gradient_absolute_value_mask(image, axis='x', sobel_ksize=3, threshold=(0, 255)):
    # Get the absolute value of the derivative
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize))
    elif axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize))
    else:
        raise 'Invalid value for axis: {}'.format(axis)

    mask = _apply_mask(sobel, threshold)
    return mask


def _gradient_magnitude_mask(image, sobel_ksize=3, threshold=(0, 255)):
    x, y = _calculate_gradients(image, sobel_ksize)
    # Calculate the magnitude
    magnit = np.sqrt(x ** 2 + y ** 2)
    mask = _apply_mask(magnit, threshold)
    return mask


def _gradient_direction_mask(image, sobel_ksize=3, threshold=(0, 255)):
    x, y = _calculate_gradients(image, sobel_ksize)
    direct = np.arctan2(np.absolute(y), np.absolute(x))
    mask = _mask_between_thresholds(direct, threshold=threshold)
    return mask


def _apply_mask(image, threshold):
    # Scale to 8 bit
    img = np.uint8(255 * image / np.max(image))
    mask = _mask_between_thresholds(img, threshold=threshold)
    return mask


def _mask_between_thresholds(img, threshold=(0, 255)):
    # Mask with 1's where the gradient magnitude is between thresholds
    mask = np.zeros_like(img)
    mask[(img >= threshold[0]) & (img <= threshold[1])] = 1
    # Return the mask as the binary image
    return mask


def _calculate_gradients(image, sobel_ksize):
    x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    return x, y


for image in glob.glob('test_images/test*.jpg'):
    image = mpimg.imread(image)
    edges = single_edges(image)
    stack = stack_edges(image)

    f, (x1, x2, x3) = plt.subplots(1, 3, figsize=(24, 9))
    x1.axis('off')
    x1.imshow(image)
    x1.set_title('Original', fontsize=20)

    x2.axis('off')
    x2.imshow(edges)
    x2.set_title('Edges', fontsize=20)

    x3.axis('off')
    x3.imshow(stack)
    x3.set_title('Stacked', fontsize=20)

