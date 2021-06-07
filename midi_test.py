from midiutil import MIDIFile
import os
import math

def cents2frequency(root_note,cent_list):
    """
    Converts each list of box parameters into MIDI format (note, duration, loudness) using the mido or pyaudio module
    """
    freq_list=[]
    for note in cent_list:
        freq_list.append(root_note*math.pow(2, note/1200.0))
    return freq_list


cwd = os.getcwd()

output_file = os.path.join(cwd,"edo31.mid")

scale_directory = os.path.join(cwd,"scales")

for scale_name in os.listdir(scale_directory):
    with open(os.path.join(scale_directory,scale_name), "r") as scale_file:
        scale_str = scale_file.read().replace('\n', ' ')

        scale_str_list = scale_str.split('!')
        scale_list_dirty = scale_str_list[-1].split(' ')

        scale_root = float(scale_str_list[2].split(' ')[-2][:-2]) #take only the Hz value at the end and remove the Hz symbol
        scale_list = [0.0]+[float(x) for x in scale_list_dirty if x]

freq_list = cents2frequency(scale_root, scale_list)

frequency_mapping = [ (i+1,note) for i, note in enumerate(freq_list)]

midi_file = MIDIFile(1, adjust_origin=False)

# Change the tuning
midi_file.changeNoteTuning(0, frequency_mapping, tuningProgam=0)

# Tell fluidsynth what bank and program to use (0 and 0, respectively)
midi_file.changeTuningBank(0, 0, 0, 0)
midi_file.changeTuningProgram(0, 0, 0, 0)

# Add some ones

for time in range(31):
    midi_file.addNote(0,0,69+time, time, 1, 100)

# Write to disk
with open(output_file, "wb") as out_file:
    midi_file.writeFile(out_file)