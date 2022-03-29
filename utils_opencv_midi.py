import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft
import math
import os
from midiutil import MIDIFile
from utils_scale import get_scale_cents_and_root, make_exp_scale_list, qbox_list2midi

def setup(file_name,scale_file_name,note_num = 8,
          beat_num = 400,root_note = 69, forced_duration = 1):

    # img=cv.imread("img.png")
    # img_thresh=threshold_test(img)

    file_name = file_name
    scale_file_name = scale_file_name

    note_num = note_num # number of pitch subdivisions
    beat_num = beat_num # number of time subdivisions in beats
    root_note = root_note #midi number corresponding to where to start. 74 starts at c5, A4 at 69
    forced_duration = forced_duration
    bpm = 120
    midi_str = file_name + '_' + scale_file_name + '_' + str(note_num) + '_' + str(beat_num) + '.mid'
    csv_str = file_name + '_' + scale_file_name + '_' + str(note_num) + '_' + str(beat_num)


    img_thresh_rgb = cv.imread(os.path.join('Images',file_name+'.png'), 0)
    img_thresh_rgb_flipped = np.flipud(img_thresh_rgb)
    img_thresh = cv.threshold(img_thresh_rgb_flipped, 127, 255, cv.THRESH_BINARY)[1]
    img_shape = img_thresh.shape

    # print("Number of foreground objects", label_count)
    # cv.imshow("Connected Components", label_image)
    # cv.waitKey(0)

    contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    box_list = get_boxes(contours)
    # note_movie(img_thresh, box_list, beat_num, bpm, forced_duration, frame_rate=1)

    img_boxes = show_boxes(img_thresh, box_list)
    #img_output_file = os.path.join(os.getcwd(),'Images','boxes','boxes_'+ midi_str + '.png')
    #cv.imwrite(img_output_file, np.flipud(img_boxes))

    scale = get_scale_cents_and_root(scale_file_name+'.scl')

    exp_scale_list = make_exp_scale_list(scale, note_num)

    qbox_list, qbox_list2, qparam_list = quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num,forced_duration=forced_duration)

    import pandas as pd
    output_file = os.path.join(os.getcwd(), 'midi_files', csv_str)
    #pd.DataFrame(qbox_list[:,0]).to_csv(output_file+'_0'+'.csv', header=None, index=None)
    #pd.DataFrame(qbox_list[:,1]).to_csv(output_file+'_1'+'.csv', header=None, index=None)
    np.flip(qbox_list[:, 0]).tofile(output_file+'_0'+'.csv',sep=',')
    np.flip(qbox_list[:, 1]).tofile(output_file+'_1'+'.csv',sep=',')

    qbox_list2midi(qbox_list,root_note, exp_scale_list, midi_str, forced_duration=forced_duration)


    img_qboxes = qshow_boxes(img_thresh, qbox_list2, qparam_list)

    qimg_output_file = os.path.join(os.getcwd(),'Images','boxes','q_boxes_'+ midi_str + '.png')
    cv.imwrite(qimg_output_file,np.flipud(img_qboxes))

    cv.imshow("Quantized box differences", img_boxes+img_qboxes)
    cv.waitKey(0)
    qimg_both_output_file = os.path.join(os.getcwd(),'Images','boxes','q_both_boxes_'+ midi_str + '.png')
    cv.imwrite(qimg_both_output_file, np.flipud(img_boxes+img_qboxes))

    # label_count, label_image = count_objects(img_thresh)
    # label_count, label_image = quantize_image(img_thresh,box_list,qbox_list)

    # box_params=get_boxes(img)
    # midi_sequence=midi_convert(box_params)

def threshold_test(img):
    """
    Test function that converts RGB image to thresholded counterpart, not really used right now
    :param img: image to be thresholded
    :return:
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_lap = cv.Laplacian(img_gray, cv.CV_64F, ksize=5)
    abs_sobel64f = np.absolute(img_lap)
    sobel_8u = np.uint8(abs_sobel64f)
    blurred = cv.blur(sobel_8u, (3, 3))
    thresh, output_binthresh = cv.threshold(blurred, 28, 255, cv.THRESH_BINARY)

    output_adapthresh = cv.adaptiveThreshold(img_gray, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, -20.0)
    cv.imshow("Adaptive Thresholding", output_adapthresh)
    cv.waitKey(0)
    return output_adapthresh


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

def note_movie(img_thresh,box_list,beat_num,bpm,duration,frame_rate=1):
    """
    writes frames to disk highlighting played notes at each moment
    :param img_thresh: thresholded image
    :return:
    """
    fpbeat = 60.0 / bpm * frame_rate
    frame_num = int(beat_num*fpbeat)
    for frame_i in range(frame_num):
        frame = np.zeros(img_thresh.shape)
        for box in box_list:
            frame_box_diff = abs(frame_i/(60.0/bpm*frame_rate)-box[1])
            if abs(frame_i/fpbeat-box[1]) < duration:
                frame[int(box[1]):int(box[1] + box[3]),int(box[0]):int(box[0] + box[2])] = img_thresh[ int(box[1]):int(box[1] + box[3]),int(box[0]):int(box[0] + box[2])]
        cv.imwrite("frame{}.png".format(frame_i),frame)
    return

def show_boxes(img_thresh, box_list, color=(255, 0, 0)):
    """
    Show a plot with boxes around each dot
    :param img_thresh:
    :param boxes:
    :return:
    """
    img_thresh_rgb = cv.cvtColor(img_thresh.copy(), cv.COLOR_GRAY2RGB)
    for box in box_list:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img_thresh_rgb, top_left, bottom_right, color, -1)
    return img_thresh_rgb

def qshow_boxes(img_thresh, qbox_list,qparam_list, color=(0, 255, 0)):
    """
    Show a plot with boxes around each dot
    :param img_thresh:
    :param boxes:
    :return:
    """
    black_img = np.zeros(img_thresh.shape, np.uint8)
    img_thresh_rgb = cv.cvtColor(black_img, cv.COLOR_GRAY2RGB)
    qp=qparam_list[0]
    qt=qparam_list[1]
    for box in qbox_list:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img_thresh_rgb, top_left, bottom_right, color, -1)
    return img_thresh_rgb

def quantize_image(img_thresh, box_list, qbox_list):
    """
    Counts the number of objects in the binary image
    :param box_list:
    :param qbox_list:
    :param img_thresh:
    :return:
    """
    box_diff_list = (box_list - qbox_list) // 1

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


def quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num,return_pixel_units=False,forced_duration=0):
    """
    Quantizes the time coordinate (y coordinate) of the picture
    :param box_list:
    :param img_height:
    :param time_divisions:
    :return:
    """
    qbox_list = np.zeros(box_list.shape)


    img_height = img_shape[0]
    img_width = img_shape[1]

    qexp_scale_list=[round(note/100) for note in exp_scale_list]

    #quantize the x (pitch) coordinate by converting it to the note index of the nearest scale note
    pitch_norm_param = img_width / qexp_scale_list[-1] # quantization parameter
    pitch_list = box_list[:, 0]
    pitch_list_normed = np.asarray(pitch_list) / pitch_norm_param
    pitch_list_normed_flat = np.repeat(pitch_list_normed, note_num)
    exp_scale_list_flat = np.tile(qexp_scale_list, len(pitch_list))
    pitch_list_normed_tile = np.reshape(pitch_list_normed_flat, (len(pitch_list), note_num))
    exp_scale_list_tile = np.reshape(exp_scale_list_flat, (len(pitch_list), note_num))
    note_ind_list = np.argmin(np.abs(pitch_list_normed_tile - exp_scale_list_tile), axis=1)

    #qbox_list[:,0] = note_ind_list
    note_list = [int(qexp_scale_list[i]) for i in note_ind_list]
    qbox_list[:, 0] = note_list
    qbox_list[:, 3] = np.asarray(box_list[:, 3]) / pitch_norm_param

    # quantize the y (time) axis
    quant_param = img_height / beat_num
    qtime_list = (box_list[:, 1] / quant_param).astype(int)
    #qtime_list = qtime_list_boxnum * int(quant_param)

    qduration_list = box_list[:, 3] / quant_param
    #qtime_list = qtime_list_boxnum * int(quant_param)

    qbox_list[:,1] = qtime_list
    qbox_list[:,2] = np.ceil(qduration_list)

    means = []
    qbox_unique = np.unique(qbox_list[:,0], return_counts=True)
    for i in reversed(np.unique(qbox_list[:,0])):
        tmp = qbox_list[np.where(qbox_list[:,0] == i)]
        tmp[:,2] = np.mean(tmp[:, 2], dtype=int)
        means.append(tmp)

    duration_mean = np.concatenate(means, axis=0)
    qbox_list[:, 2] = duration_mean[:, 2]

    #if forced_duration:
        #qbox_list[:, 2] = int(forced_duration)


    if return_pixel_units:
        qbox_list[:, 0] = note_list
        qbox_list[:, 1] = qtime_list

    qbox_list = qbox_list.astype(int)

    qparam_list = [pitch_norm_param,quant_param]
    qbox_list2 = box_list
    qbox_list2[:,0] = qbox_list[:,0]*pitch_norm_param
    qbox_list2[:,1] = qbox_list[:,1]*quant_param

    return qbox_list, qbox_list2, qparam_list


def contour2fourier(contours, n=100000,interpoints=100):
    """
    Convert contour pixel list into fourier descriptor
    """
    descriptor_list = []
    for i,contour in enumerate(contours):
        #f_contour = interp1d(contour[0], contour[1], kind='cubic')
        #contour_norm = f_contour(100)

        x= np.linspace(0.0, 2* np.pi*n)/n
        #c1 = np.sin(30.0 * 2.0 * np.pi * x) + 1j * np.sin(30.0 * 2.0 * np.pi * x)
        #c2 = np.cos(60.0 * 2.0 * np.pi * x) + 1j * np.sin(60.0 * 2.0 * np.pi * x)
        #complex_ctr = c1 + c2
        complex_ctr = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        #descriptor_complex = np.fft.fft(complex_ctr, axis=0, n=n)
        descriptor_complex = fft(complex_ctr, axis=0, n=n)
        descriptor_abs = np.abs(descriptor_complex)
        descriptor = descriptor_abs[1:] / descriptor_abs[1]
        descriptor_list.append(descriptor_abs)

        plt.title("Descriptor {} with {} pixels ".format(i,len(contour)))
        plt.plot(range(1, n), descriptor[:n])
        plt.show()

    return descriptor_list