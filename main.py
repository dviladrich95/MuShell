from utils_opencv_midi import setup

if __name__ == '__main__':
    file_name_list = [
                      'schnecke10_thresh_strip_half_size',]
                      # 'schnecke2_thresh_strip',
                      # 'schnecke6_thresh_strip_1_slanted',
                      # 'schnecke6_thresh_strip_2_slanted',
                      # 'schnecke8_thresh_strip',
                      # 'schnecke9_thresh_strip']

    scale_file_name_list = [
                            'asmaroneng_pelog']
    # 'balafon_1',
    # 'a_major_natural_equal_temperament',
    # 'a_minor_natural_equal_temperament',

    for file_name in file_name_list:
        print(file_name)
        for scale_file_name in scale_file_name_list:
            setup(file_name, scale_file_name, note_num=12,
                  beat_num=800, root_note=69)