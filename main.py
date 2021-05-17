import cv2 as cv
#import mido
from matplotlib import pyplot as plt
import numpy as np


def threshold_test(img):
    """
    :param img:
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
    :param img_thresh:
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

def quantize_image(img_thresh,box_list,qbox_list):
    """
    Counts the number of objects in the binary image
    :param img_thresh:
    :return:
    """
    box_diff_list = (box_list-qbox_list)//1

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
    box: (length,height,x_center,y_center)
    :param img_thresh:
    :return:
    """

    boxes = []
    for ctr in contours:
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append([x, y, w, h])
    boxes = np.asarray(boxes)
    return boxes

def quantize_time(box_list,img_height,time_divisions=120): #30 seconds at 120 bpm
    """

    :param box_list:
    :param img_height:
    :param time_divisions:
    :return:
    """
    quant_param= img_height/time_divisions
    qtime_list = box_list[:,1]//quant_param
    qtime_list_multiplied = qtime_list*int(quant_param)
    return qtime_list


def quantize_pitch(box_list,img_width,note_num=18): #18 because its roughly the number of
    quant_param= img_width/note_num
    qpitch_list = box_list[:,0]//quant_param

    return qpitch_list

def contour2fourier(contours,n=1000):
    """
    Convert contour pixel list into fourier descriptor
    """
    descriptor_list=[]
    for contour in contours:
        complex_ctr = contour[:,0,0] + 1j * contour[:,0,1]
        descriptor_complex=np.fft.fft(complex_ctr,axis=0,n=n)
        descriptor_abs= np.abs(descriptor_complex)
        descriptor = descriptor_abs[1:]/descriptor_abs[1]
        descriptor_list.append(descriptor_abs)

        n_list=range(1,n)
        plt.plot(n_list,descriptor)
        plt.show()

    return descriptor_list

def show_boxes(img_thresh, boxes):
    """

    :param img_thresh:
    :param boxes:
    :return:
    """
    img_thresh_rgb = cv.cvtColor(img_thresh.copy(), cv.COLOR_GRAY2RGB)
    for box in boxes:
        top_left = (box[0], box[1])
        bottom_right = (box[0] + box[2], box[1] + box[3])
        cv.rectangle(img_thresh_rgb, top_left, bottom_right, (0, 255, 0), 2)

    cv.imshow("Connected Components", img_thresh_rgb)
    cv.waitKey(0)
    return img_thresh_rgb

def boxes2midi(box_params):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness)
    """

def midi2audio(box_params):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness)
    """


if __name__ == '__main__':
    # img=cv.imread("img.png")
    # img_thresh=threshold_test(img)

    img_thresh_rgb = cv.imread("Images/shell_thresh.png", 0)
    img_thresh = cv.threshold(img_thresh_rgb, 127, 255, cv.THRESH_BINARY)[1]
    img_height, img_width = img_thresh.shape
    # print(np.alltrue(img_thresh==img_thresh_rgb))

    # print("Number of foreground objects", label_count)
    # cv.imshow("Connected Components", label_image)
    # cv.waitKey(0)

    contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    box_list = get_boxes(contours)
    descriptor_list = contour2fourier(contours)

    qtime_list=quantize_time(box_list,img_height)
    qpitch_list=quantize_pitch(box_list,img_width)
    qbox_list=zip(qpitch_list,qtime_list)


    _ = show_boxes(img_thresh, box_list)

    # label_count, label_image = count_objects(img_thresh)

    #label_count, label_image = quantize_image(img_thresh,box_list,qbox_list)

    # box_params=get_boxes(img)
    # midi_sequence=midi_convert(box_params)
