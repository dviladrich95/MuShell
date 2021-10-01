import cv2 as cv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from scipy.fft import fft
import math
import os
from midiutil import MIDIFile
from fractions import Fraction
# import pandas as pd


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
        scale_list_noconv=list(filter(None, scale_list_dirty))
        scale_list=[]
        for note in scale_list_noconv:
            if '/' in note:
                scale_list.append(math.log2(float(Fraction(note)))*1200) # numbers expressed as ratios need to be converted into cents first
        else:
            scale_list.append(float(note))

    return scale_list

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


def qbox_list2midi(qbox_list,root_note,exp_scale_list,midi_str):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness) using the MIDIutil module
    """

    output_file = os.path.join(os.getcwd(), 'midi_files', midi_str)

    tempo = 120  # In BPM
    track = 0
    channel = 0
    # duration = 1  # In beats
    volume = 100  # 0-127, as per the MIDI standard

    midi_file = MIDIFile(1, adjust_origin=False)
    midi_file.addTempo(0, 0, tempo)

    # Tell fluidsynth what bank and program to use (0 and 0, respectively)
    midi_file.changeTuningBank(0, 0, 0, 0)
    midi_file.changeTuningProgram(0, 0, 0, 0)
    # Add some notes
    for note, time, duration, _ in qbox_list:
        midi_file.addNote(track, channel, note+root_note, time, duration, volume) # time and duration measured in beats

    # Write to disk
    with open(output_file, "wb") as out_file:
        midi_file.writeFile(out_file)