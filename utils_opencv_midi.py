import cv2 as cv
import numpy as np
import os
from utils_scale import get_scale_cents_and_root, make_exp_scale_list, qbox_list2midi


def setup(file_name, scale_file_name, note_num=8, beat_num=400, root_note=69):
    file_name = file_name
    scale_file_name = scale_file_name

    note_num = note_num  # number of pitch subdivisions
    beat_num = beat_num  # number of time subdivisions in beats
    root_note = root_note  # midi number corresponding to where to start. 74 starts at c5, A4 at 69
    # bpm = 120
    midi_str = file_name + '_' + scale_file_name + '_' + str(note_num) + '_' + str(beat_num) + '.mid'
    csv_str = file_name + '_' + scale_file_name + '_' + str(note_num) + '_' + str(beat_num)

    img_thresh_rgb = cv.imread(os.path.join('Images', file_name + '.png'), 0)
    img_thresh_rgb_flipped = np.flipud(img_thresh_rgb)
    img_thresh = cv.threshold(img_thresh_rgb_flipped, 127, 255, cv.THRESH_BINARY)[1]
    img_shape = img_thresh.shape

    contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    box_list = get_boxes(contours)
    img_boxes = show_boxes(img_thresh, box_list)
    scale = get_scale_cents_and_root(scale_file_name + '.scl')
    exp_scale_list = make_exp_scale_list(scale, note_num)

    qbox_list, qbox_list2, qparam_list = quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num)

    output_file = os.path.join(os.getcwd(), 'midi_files', csv_str)
    np.flip(qbox_list[:, 0]).tofile(output_file + '_0' + '.csv', sep=',')
    np.flip(qbox_list[:, 1]).tofile(output_file + '_1' + '.csv', sep=',')

    qbox_list2midi(qbox_list, root_note, exp_scale_list, midi_str)

    img_qboxes = qshow_boxes(img_thresh, qbox_list2, qparam_list)

    qimg_output_file = os.path.join(os.getcwd(), 'Images', 'boxes', 'q_boxes_' + midi_str + '.png')
    cv.imwrite(qimg_output_file, np.flipud(img_qboxes))

    cv.imshow("Quantized box differences", img_boxes + img_qboxes)
    cv.waitKey(0)
    qimg_both_output_file = os.path.join(os.getcwd(), 'Images', 'boxes', 'q_both_boxes_' + midi_str + '.png')
    cv.imwrite(qimg_both_output_file, np.flipud(img_boxes + img_qboxes))


def count_objects(img_thresh):
    """
    Counts the number of objects in the binary image
    :param img_thresh: thresholded image
    :return:
    """
    label_image = img_thresh.copy()
    label_count = 0
    print(label_image.shape)
    rows, cols = label_image.shape
    for j in range(rows):
        for i in range(cols):
            pixel = label_image[j, i]
            if 255 == pixel:
                label_count += 1
                cv.floodFill(label_image, None, (i, j), label_count)

    return label_count, label_image


def show_boxes(img_thresh, box_list, color=(255, 0, 0)):
    """
    Show a plot with boxes around each dot
    :param img_thresh:
    :param box_list:
    :param color:
    :return:
    """
    img_thresh_rgb = cv.cvtColor(img_thresh.copy(), cv.COLOR_GRAY2RGB)
    for box in box_list:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img_thresh_rgb, top_left, bottom_right, color, -1)
    return img_thresh_rgb


def qshow_boxes(img_thresh, qbox_list, qparam_list, color=(0, 255, 0)):
    """
    Show a plot with boxes around each dot
    :param img_thresh:
    :param qbox_list:
    :param qparam_list:
    :param color:
    :return:
    """
    black_img = np.zeros(img_thresh.shape, np.uint8)
    img_thresh_rgb = cv.cvtColor(black_img, cv.COLOR_GRAY2RGB)
    for box in qbox_list:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img_thresh_rgb, top_left, bottom_right, color, -1)
    return img_thresh_rgb


def get_boxes(contours):
    """
    Counts the number of points in the image and returns a list of the fitting box parameters associated with each
    box: (x_center,y_center,width,height)
    :param contours:
    :return:
    """

    box_list = []
    for ctr in contours:
        x, y, w, h = cv.boundingRect(ctr)
        box_list.append([x, y, w, h])
    box_list = np.asarray(box_list)
    return box_list


def quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num, return_pixel_units=False):
    """
    Quantizes the time coordinate (y coordinate) of the picture
    :param box_list:
    :param img_shape:
    :param exp_scale_list:
    :param note_num:
    :param beat_num:
    :param return_pixel_units:
    :return:
    """

    qbox_list = np.zeros(box_list.shape)

    img_height = img_shape[0]
    img_width = img_shape[1]

    qexp_scale_list = [round(note / 100) for note in exp_scale_list]

    # quantize the x (pitch) coordinate by converting it to the note index of the nearest scale note
    pitch_norm_param = img_width / qexp_scale_list[-1]  # quantization parameter
    pitch_list = box_list[:, 0]
    pitch_list_normed = np.asarray(pitch_list) / pitch_norm_param
    pitch_list_normed_flat = np.repeat(pitch_list_normed, note_num)
    exp_scale_list_flat = np.tile(qexp_scale_list, len(pitch_list))
    pitch_list_normed_tile = np.reshape(pitch_list_normed_flat, (len(pitch_list), note_num))
    exp_scale_list_tile = np.reshape(exp_scale_list_flat, (len(pitch_list), note_num))
    note_ind_list = np.argmin(np.abs(pitch_list_normed_tile - exp_scale_list_tile), axis=1)

    # qbox_list[:,0] = note_ind_list
    note_list = [int(qexp_scale_list[i]) for i in note_ind_list]
    qbox_list[:, 0] = note_list
    qbox_list[:, 3] = np.asarray(box_list[:, 3]) / pitch_norm_param

    # quantize the y (time) axis
    quant_param = img_height / beat_num
    qtime_list = (box_list[:, 1] / quant_param).astype(int)

    qduration_list = box_list[:, 3] / quant_param

    qbox_list[:, 1] = qtime_list
    qbox_list[:, 2] = np.ceil(qduration_list)

    means = []
    for i in reversed(np.unique(qbox_list[:, 0])):
        tmp = qbox_list[np.where(qbox_list[:, 0] == i)]
        tmp[:, 2] = np.mean(tmp[:, 2], dtype=int)
        means.append(tmp)

    duration_mean = np.concatenate(means, axis=0)
    qbox_list[:, 2] = duration_mean[:, 2]

    if return_pixel_units:
        qbox_list[:, 0] = note_list
        qbox_list[:, 1] = qtime_list

    qbox_list = qbox_list.astype(int)

    qparam_list = [pitch_norm_param, quant_param]
    qbox_list2 = box_list
    qbox_list2[:, 0] = qbox_list[:, 0] * pitch_norm_param
    qbox_list2[:, 1] = qbox_list[:, 1] * quant_param

    return qbox_list, qbox_list2, qparam_list
