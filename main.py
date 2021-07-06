import cv2 as cv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft
import math
import os
from midiutil import MIDIFile

# import pandas as pd

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

def show_boxes(img_thresh, box_list, color):
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

    cv.imshow("Connected Components", img_thresh_rgb)
    cv.waitKey(0)
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



def make_exp_scale_list(scale, note_num):
    """

    :param scale: scale to use in cents
    :param note_num: number of notes to use, the scale repeates until all notes in the range are used
    :return: list of notes using the given scale
    """
    scale_range = scale[-1] # range of the scale, will always be 1200 for scales that span a whole octave
    exp_scale_list=[0.0] # initial note was not part of the scale, is added now
    for i in range(note_num-1):
        exp_scale_list.append(scale[i%len(scale)]+i//(len(scale))*scale_range)
    return exp_scale_list


def quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num=120,return_pixel_units=False):  # 30 seconds at 120 bpm
    """
    Quantizes the time coordinate (y coordinate) of the picture
    :param box_list:
    :param img_height:
    :param time_divisions:
    :return:
    """
    qbox_list = np.zeros(box_list.shape)

    img_width = img_shape[0]
    img_height = img_shape[1]

    #quantize the x (pitch) coordinate by converting it to the note index of the nearest scale note
    pitch_norm_param = img_width / exp_scale_list[-1] # quantization parameter
    pitch_list = box_list[:, 0]
    pitch_list_normed = np.asarray(pitch_list) / pitch_norm_param
    pitch_list_normed_flat = np.repeat(pitch_list_normed, note_num)
    exp_scale_list_flat = np.tile(exp_scale_list, len(pitch_list))
    pitch_list_normed_tile = np.reshape(pitch_list_normed_flat, (len(pitch_list), note_num))
    exp_scale_list_tile = np.reshape(exp_scale_list_flat, (len(pitch_list), note_num))
    note_ind_list = np.argmin(np.abs(pitch_list_normed_tile - exp_scale_list_tile), axis=1)

    qbox_list[:,0] = note_ind_list
    note_list = [exp_scale_list[i] for i in note_ind_list]

    # quantize the
    quant_param = img_height / beat_num
    qtime_list = (box_list[:, 1] / quant_param).astype(int)
    #qtime_list = qtime_list_boxnum * int(quant_param)

    #TODO implement duration
    quant_param = img_height / beat_num
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

    if return_pixel_units:
        qbox_list[:, 0] = qtime_list
        qbox_list[:, 1] = qtime_list

    return qbox_list.astype(int)

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


def get_scale_paths():
    cwd = os.getcwd()

    scale_directory = os.path.join(cwd, "scales")
    scales_name_list = os.listdir(scale_directory)
    return scales_name_list

def get_scale_cents_and_root(scale_name):
    """
    Gets the scale file and extracts a list of note coordinates in cents
    :param scale_name: name of the scale file to be used
    :return: root node of the sclae and scale note list
    """
    cwd = os.getcwd()
    scale_directory = os.path.join(cwd, "scales")
    with open(os.path.join(scale_directory,scale_name), "r") as scale_file:
        scale_str = scale_file.read().replace('\n', ' ')

        scale_str_list = scale_str.split('!')
        scale_list_dirty = scale_str_list[-1].split(' ')

        scale_root = float(scale_str_list[2].split(' ')[-2][:-2]) #take only the Hz value at the end and remove the Hz symbol
        scale_list = [float(x) for x in scale_list_dirty if x]
    return scale_root, scale_list

def cents2frequency(cent_list,root_note):
    """
    Converts each list of notes in cents into frequency given root node
    :param cent_list: list of notes in cents
    :param root_note: root node that will indicate the frequency of the first note with 0 cents
    :return: list of notes in frequency (Hz)
    """
    freq_list = []
    for note in cent_list:
        freq_list.append(root_note*math.pow(2, note/1200.0))
    return freq_list

def qbox_list2midi(qbox_list,root_note,exp_scale_list,midi_str):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness) using the MIDIutil module
    """

    output_file = os.path.join(os.getcwd(), 'midi_files', midi_str)
    freq_list = cents2frequency(exp_scale_list, root_note)

    frequency_mapping = [(i, note) for i, note in enumerate(freq_list)]

    tempo = 120  # In BPM
    track = 0
    channel = 0
    # duration = 1  # In beats
    volume = 100  # 0-127, as per the MIDI standard

    midi_file = MIDIFile(1, adjust_origin=False)
    midi_file.addTempo(0, 0, tempo)

    # Change the tuning
    midi_file.changeNoteTuning(0, frequency_mapping, tuningProgam=0)

    # Tell fluidsynth what bank and program to use (0 and 0, respectively)
    midi_file.changeTuningBank(0, 0, 0, 0)
    midi_file.changeTuningProgram(0, 0, 0, 0)
    # Add some notes
    for note_ind, time, duration, _ in qbox_list:
        midi_file.addNote(track, channel, note_ind, time, duration, volume) # time and duration measured in beats

    # Write to disk
    with open(output_file, "wb") as out_file:
        midi_file.writeFile(out_file)



if __name__ == '__main__':
    # img=cv.imread("img.png")
    # img_thresh=threshold_test(img)
    file_name = "nussatella_tresh_1"
    scale_file_name = '12_ed_2_equal_temperament'
    midi_str= file_name + scale_file_name +'.mid'
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

    root_note, scale = get_scale_cents_and_root(scale_file_name+'.scl')
    note_num=16
    exp_scale_list = make_exp_scale_list(scale, note_num)

    qbox_list = quantize_box(box_list, img_shape, exp_scale_list, note_num, beat_num=8)


    qbox_list2midi(qbox_list,root_note,exp_scale_list, midi_str)

    #img_boxes = show_boxes(img_thresh, box_list, color=(0, 255, 0))
    #img_qboxes = show_boxes(img_thresh, qbox_list, color=(255, 0, 0))

    #cv.imshow("Quantized box differences", img_boxes+img_qboxes)
    #cv.waitKey(0)

    # label_count, label_image = count_objects(img_thresh)
    # label_count, label_image = quantize_image(img_thresh,box_list,qbox_list)

    # box_params=get_boxes(img)
    # midi_sequence=midi_convert(box_params)
