import cv2 as cv
#import mido
from matplotlib import pyplot as plt
import numpy as np


def threshold_test(shell):
    """
    :param shell:
    :return:

    """
    shell_gray = cv.cvtColor(shell, cv.COLOR_BGR2GRAY)
    shell_lap = cv.Laplacian(shell_gray, cv.CV_64F, ksize=5)
    abs_sobel64f = np.absolute(shell_lap)
    sobel_8u = np.uint8(abs_sobel64f)
    blurred = cv.blur(sobel_8u, (3, 3))
    thresh, output_binthresh = cv.threshold(blurred, 28, 255, cv.THRESH_BINARY)

    output_adapthresh = cv.adaptiveThreshold(shell_gray, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, -20.0)
    cv.imshow("Adaptive Thresholding", output_adapthresh)
    cv.waitKey(0)
    return output_adapthresh


def count_objects(shell_thresh):
    """
    Counts the number of objects in the binary image
    :param shell_thresh:
    :return:
    """
    label_image = shell_thresh.copy()
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


def get_boxes(shell_thresh):
    """
    Counts the number of points in the image and returns a list of the fitting box parameters associated with each
    box: (length,height,x_center,y_center)
    :param shell_thresh:
    :return:
    """

    boxes = []
    for ctr in ctrs:
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append([x, y, w, h])
    return boxes

def contour2fourier(contours,n=10):
    """
    Convert contour pixel list into fourier descriptor
    """
    descriptor_list=[]
    for contour in contours:
        complex_ctr = contour[:,0,0] + 1j * contour[:,0,1]
        descriptor_complex=np.fft.fft(complex_ctr,axis=0,n=n)
        descriptor_abs= np.abs(descriptor_complex)
        descriptor_list.append(descriptor_abs)

        n_list=range(1,n)
        plt.plot(n_list,descriptor_abs[1:])
        plt.show()




    return descriptor_list

def show_boxes(shell_thresh, boxes):
    """

    :param shell_thresh:
    :param boxes:
    :return:
    """
    shell_thresh_rgb = cv.cvtColor(shell_thresh.copy(), cv.COLOR_GRAY2RGB)
    for box in boxes:
        top_left = (box[0], box[1])
        bottom_right = (box[0] + box[2], box[1] + box[3])
        cv.rectangle(shell_thresh_rgb, top_left, bottom_right, (0, 255, 0), 2)

    cv.imshow("Connected Components", shell_thresh_rgb)
    cv.waitKey(0)
    return shell_thresh_rgb

def boxes2midi(box_params):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness)
    """


def midi2audio(box_params):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness)
    """


if __name__ == '__main__':
    # shell=cv.imread("shell.png")
    # shell_thresh=threshold_test(shell)

    shell_thresh_rgb = cv.imread("Images/ellipse.jpg", 0)
    shell_thresh = cv.threshold(shell_thresh_rgb, 127, 255, cv.THRESH_BINARY)[1]

    # print(np.alltrue(shell_thresh==shell_thresh_rgb))
    # label_count, label_image = count_objects(shell_thresh)
    # print("Number of foreground objects", label_count)
    # cv.imshow("Connected Components", label_image)
    # cv.waitKey(0)

    ctrs, _ = cv.findContours(shell_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    descriptor_list = contour2fourier(ctrs)
    boxes = get_boxes(ctrs)

    _ = show_boxes(shell_thresh,boxes)

    # box_params=get_boxes(img)
    # midi_sequence=midi_convert(box_params)
