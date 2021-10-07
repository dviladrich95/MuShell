import cv2 as cv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft
import math
import os
from midiutil import MIDIFile
# import pandas as pd

from utils_opencv_midi import count_objects, show_boxes, qshow_boxes, quantize_image, get_boxes, quantize_box, contour2fourier
from utils_scale import get_scale_cents_and_root, make_exp_scale_list, qbox_list2midi




if __name__ == '__main__':
    # img=cv.imread("img.png")
    # img_thresh=threshold_test(img)

    file_name = "conus_textile_lightbox_thresh_strip_grid_1"
    scale_file_name = 'ionic'

    note_num= 60 # number of pitch subdivisions
    beat_num = 180 # number of time subdivisions in beats
    midi_str = file_name + '_' + scale_file_name + '_' + str(note_num) + '_' + str(beat_num) + '.mid'
    root_note = 0 #midi number corresponding to where to start. 74 starts at c5

    img_thresh_rgb = cv.imread(os.path.join('Images',file_name+'.png'), 0)
    img_thresh = cv.threshold(img_thresh_rgb, 127, 255, cv.THRESH_BINARY)[1]
    img_shape = img_thresh.shape
    # print(np.alltrue(img_thresh==img_thresh_rgb))

    # print("Number of foreground objects", label_count)
    # cv.imshow("Connected Components", label_image)
    # cv.waitKey(0)

    contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    box_list = get_boxes(contours)
    #descriptor_list = contour2fourier(contours)

    scale = get_scale_cents_and_root(scale_file_name+'.scl')

    exp_scale_list = make_exp_scale_list(scale, note_num)

    qbox_list, qparam_list = quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num)


    qbox_list2midi(qbox_list,root_note, exp_scale_list, midi_str)

    img_boxes = show_boxes(img_thresh, box_list)
    img_qboxes = qshow_boxes(img_thresh, qbox_list, qparam_list, color=(255, 0, 0))

    cv.imshow("Quantized box differences", img_boxes+img_qboxes)
    cv.waitKey(0)

    # label_count, label_image = count_objects(img_thresh)
    # label_count, label_image = quantize_image(img_thresh,box_list,qbox_list)

    # box_params=get_boxes(img)
    # midi_sequence=midi_convert(box_params)
