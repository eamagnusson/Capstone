# From arjunsharma97

# Packages reqd: pydub, ffmpeg

# pydub - pip install pydub

# ffmpeg: 
# sudo add-apt-repository ppa:kirillshkrogalev/ffmpeg-next
# sudo apt-get update
# sudo apt-get install ffmpeg

## Load the m4a files (in M4a_files.tar.gz) 

import os
import argparse

from pydub import AudioSegment

formats_to_convert = ['.m4a']

# Place folder pathname here -->
<<<<<<< HEAD
for (dirpath, dirnames, filenames) in os.walk("/Users/evanmagnusson/sysCapstone/Capstone/0"):
=======
for (dirpath, dirnames, filenames) in os.walk("/Users/evanmagnusson/sysCapstone/Capstone/0/M4a_files"):
>>>>>>> parent of 65d4bb6 (Slight fixes)
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):

            filepath = dirpath + '/' + filename
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath,
                        file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                print("ERROR CONVERTING " + str(filepath))