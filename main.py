import cv2 as cv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft
import math
import os
from midiutil import MIDIFile
# import pandas as pd

from utils_opencv_midi import setup, count_objects, show_boxes, qshow_boxes, quantize_image, get_boxes, quantize_box, contour2fourier
from utils_scale import get_scale_cents_and_root, make_exp_scale_list, qbox_list2midi




if __name__ == '__main__':
    #'conus_textile_lightbox_thresh_strip_dots',
    file_name_list = [
                      'schnecke10_thresh_strip_half_size',]
                      # 'schnecke2_thresh_strip',
                      # 'schnecke6_thresh_strip_1_slanted',
                      # 'schnecke6_thresh_strip_2_slanted',
                      # 'schnecke8_thresh_strip',
                      # 'schnecke9_thresh_strip']

    # 'balafon_1',
    # 'a_major_natural_equal_temperament',
    # 'a_minor_natural_equal_temperament',
    # 'asmaroneng_pelog',
    scale_file_name_list = [
                            'asmaroneng_pelog']

    # setup('schnecke1_thresh_strip', 'balafon_1', note_num=12,
    #       beat_num=400, root_note=69, forced_duration=1)

    for file_name in file_name_list:
        print(file_name)
        for scale_file_name in scale_file_name_list:
            setup(file_name, scale_file_name, note_num=12,
                  beat_num=800, root_note=69, forced_duration=1)