import warnings
import cv2
import numpy as np
import imutils
import argparse
from .frame import Frame
import math


def layer_overlay(foreground, background):
    """put an image over the top of another

    Intended for use in VidStab class to create a trail of previous frames in the stable video output.

    :param foreground: image to be laid over top of background image
    :param background: image to over laid with foreground image
    :return: return combined image where foreground is laid over background

    >>> from vidstab import VidStab, layer_overlay, layer_blend
    >>>
    >>> stabilizer = VidStab()
    >>>
    >>> stabilizer.stabilize(input_path='my_shaky_video.avi',
    >>>                      output_path='stabilized_output.avi',
    >>>                      border_size=100,
    >>>                      layer_func=layer_overlay)
    """
    overlaid = foreground.copy()
    negative_space = np.where(foreground[:, :, 3] == 0)

    overlaid[negative_space] = background[negative_space]

    overlaid[:, :, 3] = 255

    return overlaid


def layer_blend(foreground, background, foreground_alpha=.6):
    """blend a foreground image over background (wrapper for cv2.addWeighted)

    :param foreground: image to be laid over top of background image
    :param background: image to over laid with foreground image
    :param foreground_alpha: alpha to apply to foreground; (1 - foreground_alpha) applied to background
    :return: return combined image where foreground is laid over background with alpha

    >>> from vidstab import VidStab, layer_overlay, layer_blend
    >>>
    >>> stabilizer = VidStab()
    >>>
    >>> stabilizer.stabilize(input_path='my_shaky_video.avi',
    >>>                      output_path='stabilized_output.avi',
    >>>                      border_size=100,
    >>>                      layer_func=layer_blend)
    """
    cv2.addWeighted(foreground, foreground_alpha,
                    background, 1 - foreground_alpha, 0, background)

    return background


def apply_layer_func(cur_frame, prev_frame, layer_func):
    """helper method to apply layering function in vidstab process

    :param cur_frame: current frame to apply layer over prev_frame
    :param prev_frame: previous frame to be layered over by cur_frame
    :param layer_func: layering function to apply
    :return: tuple of (layered_frames, prev_frame) where prev_frame is to be used in next layering operation
    """
    if prev_frame is not None:
        cur_frame_image = layer_func(cur_frame.image, prev_frame.image)
        cur_frame = Frame(cur_frame_image, color_format=cur_frame.color_format)

    return cur_frame



def functional_border_sizes(border_size):
    """Calculate border sizing used in process to gen user specified border size

    If border_size is negative then a stand-in border size is used to allow better keypoint tracking (i think... ?);
    negative border is then applied at end.

    :param border_size: user supplied border size
    :return: (border_size, neg_border_size) tuple of functional border sizes

    >>> functional_border_sizes(100)
    (100, 0)
    >>> functional_border_sizes(-10)
    (100, 110)
    """
    if border_size < 0:
        neg_border_size = 100 + abs(border_size)
        border_size = 100
    else:
        neg_border_size = 0

    return border_size, neg_border_size


def crop_frame(frame, border_options):
    """Handle frame cropping for auto border size and negative border size

    if auto_border is False and neg_border_size == 0 then frame is returned as is

    :param frame: frame to be cropped
    :param border_options: dictionary of border options including keys for:
        * 'border_size': functional border size determined by functional_border_sizes
        * 'neg_border_size': functional negative border size determined by functional_border_sizes
        * 'extreme_frame_corners': VidStab.extreme_frame_corners attribute
        * 'auto_border': VidStab.auto_border_flag attribute
    :return: cropped frame
    """
    if not border_options['auto_border_flag'] and border_options['neg_border_size'] == 0:
        return frame

    if border_options['auto_border_flag']:
        cropped_frame_image = auto_border_crop(frame.image,
                                               border_options['extreme_frame_corners'],
                                               border_options['border_size'])

    else:
        frame_h, frame_w = frame.image.shape[:2]
        cropped_frame_image = frame.image[
                              border_options['neg_border_size']:(frame_h - border_options['neg_border_size']),
                              border_options['neg_border_size']:(frame_w - border_options['neg_border_size'])
                              ]

    cropped_frame = Frame(cropped_frame_image, color_format=frame.color_format)

    return cropped_frame


def safe_import_cv2():
    """Gracefully fail ModuleNotFoundError due to cv2

    :return: None
    """
    try:
        import cv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("""
        No python bindings for OpenCV found when attempting to `import cv2`.
        If you have not installed OpenCV you can install with:

            pip install vidstab[cv2]

        If you'd prefer to install OpenCV from source you can see the docs here:
            https://docs.opencv.org/3.4.1/da/df6/tutorial_py_table_of_contents_setup.html
        """)


# noinspection PyPep8Naming
def cv2_estimateRigidTransform(from_pts, to_pts, full=False):
    """Estimate transforms in OpenCV 3 or OpenCV 4"""
    if not from_pts.shape[0] or not to_pts.shape[0]:
        return None

    if imutils.is_cv4():
        transform = cv2.estimateAffinePartial2D(from_pts, to_pts)[0]
    else:
        # noinspection PyUnresolvedReferences
        transform = cv2.estimateRigidTransform(from_pts, to_pts, full)

    return transform




def playback_video(display_frame, playback_flag, delay, max_display_width=750):
    if not playback_flag:
        return False

    if display_frame.shape[1] > max_display_width:
        display_frame = imutils.resize(display_frame, width=max_display_width)

    cv2.imshow('VidStab Playback ({} frame delay if using live video;'
               ' press Q or ESC to quit)'.format(delay),
               display_frame)
    key = cv2.waitKey(1)

    if key == ord("q") or key == 27:
        return True

def bfill_rolling_mean(arr, n=30):
    """Helper to perform trajectory smoothing

    :param arr: Numpy array of frame trajectory to be smoothed
    :param n: window size for rolling mean
    :return: smoothed input arr

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> bfill_rolling_mean(arr, n=2)
    array([[2.5, 3.5, 4.5],
           [2.5, 3.5, 4.5]])
    """
    if arr.shape[0] < n:
        raise ValueError('arr.shape[0] cannot be less than n')
    if n == 1:
        return arr

    pre_buffer = np.zeros(3).reshape(1, 3)
    post_buffer = np.zeros(3 * n).reshape(n, 3)
    arr_cumsum = np.cumsum(np.vstack((pre_buffer, arr, post_buffer)), axis=0)
    buffer_roll_mean = (arr_cumsum[n:, :] - arr_cumsum[:-n, :]) / float(n)
    trunc_roll_mean = buffer_roll_mean[:-n, ]

    bfill_size = arr.shape[0] - trunc_roll_mean.shape[0]
    bfill = np.tile(trunc_roll_mean[0, :], (bfill_size, 1))

    return np.vstack((bfill, trunc_roll_mean))


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def border_frame(frame, border_size, border_type):
    """Convenience wrapper of cv2.copyMakeBorder for how vidstab applies borders

    :param frame: frame to apply border to
    :param border_size: int border size in number of pixels
    :param border_type: one of the following ['black', 'reflect', 'replicate']
    :return: bordered version of frame with alpha layer for frame overlay options
    """
    border_modes = {'black': cv2.BORDER_CONSTANT,
                    'reflect': cv2.BORDER_REFLECT,
                    'replicate': cv2.BORDER_REPLICATE}
    border_mode = border_modes[border_type]

    bordered_frame_image = cv2.copyMakeBorder(frame.image,
                                              top=border_size,
                                              bottom=border_size,
                                              left=border_size,
                                              right=border_size,
                                              borderType=border_mode,
                                              value=[0, 0, 0])

    bordered_frame = Frame(bordered_frame_image, color_format=frame.color_format)

    alpha_bordered_frame = bordered_frame.bgra_image
    alpha_bordered_frame[:, :, 3] = 0
    h, w = frame.image.shape[:2]
    alpha_bordered_frame[border_size:border_size + h, border_size:border_size + w, 3] = 255

    return alpha_bordered_frame, border_mode


def match_keypoints(optical_flow, prev_kps):
    """Match optical flow keypoints

    :param optical_flow: output of cv2.calcOpticalFlowPyrLK
    :param prev_kps: keypoints that were passed to cv2.calcOpticalFlowPyrLK to create optical_flow
    :return: tuple of (cur_matched_kp, prev_matched_kp)
    """
    cur_kps, status, err = optical_flow

    # storage for keypoints with status 1
    prev_matched_kp = []
    cur_matched_kp = []

    if status is None:
        return cur_matched_kp, prev_matched_kp

    for i, matched in enumerate(status):
        # store coords of keypoints that appear in both
        if matched:
            prev_matched_kp.append(prev_kps[i])
            cur_matched_kp.append(cur_kps[i])

    return cur_matched_kp, prev_matched_kp


def estimate_partial_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process

    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    cur_matched_kp, prev_matched_kp = matched_keypoints

    transform = cv2_estimateRigidTransform(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp),
                                           False)
    if transform is not None:
        # translation x
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]


def transform_frame(frame, transform, border_size, border_type):
    if border_type not in ['black', 'reflect', 'replicate']:
        raise ValueError('Invalid border type')

    transform = build_transformation_matrix(transform)
    bordered_frame_image, border_mode = border_frame(frame, border_size, border_type)

    h, w = bordered_frame_image.shape[:2]
    transformed_frame_image = cv2.warpAffine(bordered_frame_image, transform, (w, h), borderMode=border_mode)

    transformed_frame = Frame(transformed_frame_image, color_format='BGRA')

    return transformed_frame


def post_process_transformed_frame(transformed_frame, border_options, layer_options):
    cropped_frame = crop_frame(transformed_frame, border_options)

    if layer_options['layer_func'] is not None:
        cropped_frame = apply_layer_func(cropped_frame,
                                                     layer_options['prev_frame'],
                                                     layer_options['layer_func'])

        layer_options['prev_frame'] = cropped_frame

    return cropped_frame, layer_options



def extreme_corners(frame, transforms):
    """Calculate max drift of each frame corner caused by stabilizing transforms

    :param frame: frame from video being stabilized
    :param transforms: VidStab transforms attribute
    :return: dictionary of most extreme x and y values caused by transformations
    """
    h, w = frame.shape[:2]
    frame_corners = np.array([[0, 0],  # top left
                              [0, h - 1],  # bottom left
                              [w - 1, 0],  # top right
                              [w - 1, h - 1]],  # bottom right
                             dtype='float32')
    frame_corners = np.array([frame_corners])

    min_x = min_y = max_x = max_y = 0
    for i in range(transforms.shape[0]):
        transform = transforms[i, :]
        transform_mat = vidstab_utils.build_transformation_matrix(transform)
        transformed_frame_corners = cv2.transform(frame_corners, transform_mat)

        delta_corners = transformed_frame_corners - frame_corners

        delta_y_corners = delta_corners[0][:, 1].tolist()
        delta_x_corners = delta_corners[0][:, 0].tolist()
        min_x = min([min_x] + delta_x_corners)
        min_y = min([min_y] + delta_y_corners)
        max_x = max([max_x] + delta_x_corners)
        max_y = max([max_y] + delta_y_corners)

    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}


def auto_border_start(min_corner_point, border_size):
    """Determine upper-right corner coords for auto border crop

    :param min_corner_point: extreme corner component either 'min_x' or 'min_y'
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: adjusted extreme corner for cropping
    """
    return math.floor(border_size - abs(min_corner_point))


def auto_border_length(frame_dim, extreme_corner, border_size):
    """Determine height/width auto border crop

    :param frame_dim: height/width of frame to be auto border cropped (corresponds to extreme_corner)
    :param extreme_corner: extreme corner component either 'min_x' or 'min_y' (corresponds to frame_dim)
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: adjusted extreme corner for cropping
    """
    return math.ceil(frame_dim - (border_size - extreme_corner))


def auto_border_crop(frame, extreme_frame_corners, border_size):
    """Crop frame for auto border in vidstab process

    :param frame: frame to be cropped
    :param extreme_frame_corners: extreme_frame_corners attribute of vidstab object
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: cropped frame determined by auto border process
    """
    if border_size == 0:
        return frame

    frame_h, frame_w = frame.shape[:2]

    x = auto_border_start(extreme_frame_corners['min_x'], border_size)
    y = auto_border_start(extreme_frame_corners['min_y'], border_size)

    w = auto_border_length(frame_w, extreme_frame_corners['max_x'], border_size)
    h = auto_border_length(frame_h, extreme_frame_corners['max_y'], border_size)

    return frame[y:h, x:w]


def min_auto_border_size(extreme_frame_corners):
    """Calc minimum border size to accommodate most extreme transforms

    :param extreme_frame_corners: extreme_frame_corners attribute of vidstab object
    :return: minimum border size as int
    """
    abs_extreme_corners = [abs(x) for x in extreme_frame_corners.values()]
    return math.ceil(max(abs_extreme_corners))



def str_int(v):
    """Handle argparse inputs to that could be str or int

    :param v: value to convert
    :return: v as int if int(v) does not raise ValueError

    >>> str_int('test')
    test
    >>> str_int(1)
    1
    """
    try:
        int_v = int(v)
        return int_v
    except ValueError:
        return v


def str_2_bool(v):
    """Convert string to bool from different possible strings

    :param v: value to convert
    :return: string converted to bool

    >>> str_2_bool('y')
    True
    >>> str_2_bool('0')
    False
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_max_frames_arg(max_frames_arg):
    """Handle maxFrames arg in vidstab.__main__

    Convert negative values to inf

    :param max_frames_arg: maxFrames arg in vidstab.__main__
    :return: max_frames as is or inf

    >>> process_max_frames_arg(-1)
    inf
    >>> process_max_frames_arg(1)
    1
    """
    if max_frames_arg > 0:
        return max_frames_arg
    return float('inf')


def process_layer_frames_arg(layer_frames_arg):
    """Handle layerFrames arg in vidstab.__main__

    :param layer_frames_arg: layerFrames arg in vidstab.__main__
    :return: vidstab.layer_overlay if True else None

    >>> process_layer_frames_arg(False)

    """
    if layer_frames_arg:
        return layer_overlay
    return None


def process_border_size_arg(border_size_arg):
    """Handle borderSize arg in vidstab.__main__

    Convert strings that aren't 'auto' to 0

    :param border_size_arg: borderSize arg in vidstab.__main__
    :return: int or 'auto'

    >>> process_border_size_arg('a')
    0
    >>> process_border_size_arg('auto')
    'auto'
    >>> process_border_size_arg(0)
    0
    """
    if isinstance(border_size_arg, str):
        if not border_size_arg == 'auto':
            warnings.warn('Invalid borderSize provided; converting to 0.')
            border_size_arg = 0

    return border_size_arg


def cli_stabilizer(args):
    """Handle CLI vidstab processing

    :param args: result of vars(ap.parse_args()) from vidstab.__main__
    :return: None
    """

    max_frames = process_max_frames_arg(args['maxFrames'])
    border_size = process_border_size_arg(args['borderSize'])
    layer_func = process_layer_frames_arg(args['layerFrames'])

    # init stabilizer with user specified keypoint detector
    stabilizer = VidStab(kp_method=args['keyPointMethod'].upper())
    # stabilize input video and write to specified output file
    stabilizer.stabilize(input_path=args['input'],
                         output_path=args['output'],
                         smoothing_window=args['smoothWindow'],
                         max_frames=max_frames,
                         border_type=args['borderType'],
                         border_size=border_size,
                         layer_func=layer_func,
                         playback=args['playback'])
