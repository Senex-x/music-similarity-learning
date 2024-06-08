from os import listdir
from os.path import dirname
from os.path import join
from pathlib import Path

import librosa
import librosa.display
import numpy
from pydub import AudioSegment
from pylab import *

from compression_utils import resize_matrix
from concurrency_utils import ConcurrencyUtils


class MusicTokenizer:

    def __init__(self):
        self.mfcc_length = 500
        self.music_folder_path = join(dirname(__file__), 'data/music_mp3')
        self.converted_music_folder_path = join(dirname(__file__), 'data/music_wav')
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')
        self.spectrograms_music_folder_path = join(dirname(__file__), 'data/music_spectrograms')
        self.uploaded_music_folder_path = join(dirname(__file__), 'data/uploads')

    def tokenize(self):
        self.__convert_mp3_files_to_wav(source_folder=self.music_folder_path)
        self.__extract_feature_vectors()

    def tokenize_uploaded_track(self):
        self.__convert_mp3_files_to_wav(source_folder=self.uploaded_music_folder_path)
        self.__extract_feature_vector_from_file(wav_file_name=listdir(self.uploaded_music_folder_path)[0][:-4] + '.wav')

    def __convert_mp3_files_to_wav(self, source_folder):
        def job(i, mp3_file_name, file_count):
            if not self.__track_is_converted(mp3_file_name):
                sound = AudioSegment.from_mp3(join(source_folder, mp3_file_name))
                sound.export(out_f=join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav'),
                             format="wav",
                             codec="pcm_u8",
                             parameters=["-ac", "1", "-ar", "4000"])
            print(f'Converted {i + 1} out of {file_count} .mp3 files '
                  f'({round((i + 1) / file_count * 100, 2)}%)')

        ConcurrencyUtils.process_files_in_parallel(job, source_folder)

    def __track_is_converted(self, mp3_file_name):
        return Path(join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav')).exists()

    def __extract_feature_vectors(self):
        def job(i, wav_file_name, file_count):
            self.__extract_feature_vector_from_file(wav_file_name)

            print(f'Tokenized {i + 1} out of {file_count} .wav files '
                  f'({round((i + 1) / file_count * 100, 2)}%)')

        ConcurrencyUtils.process_files_in_parallel(job, self.converted_music_folder_path)

    def __extract_feature_vector_from_file(self, wav_file_name):
        if not self.__track_is_tokenized(wav_file_name):
            try:
                x, sample_rate = librosa.load(join(self.converted_music_folder_path, wav_file_name),
                                              offset=5.0,
                                              mono=True,
                                              res_type='kaiser_fast')

                # extract features from the audio
                mfcc_matrix = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=32)
                # output: coefficients to time frames
                mfcc_matrix = resize_matrix(mfcc_matrix, target_length=self.mfcc_length)

                mfcc = np.mean(mfcc_matrix, axis=0)
                numpy.save(join(self.tokenized_music_folder_path, wav_file_name[:-4]), mfcc)
            except:
                print(f"Error while tokenizing {wav_file_name}")
                # remove(join(self.converted_music_folder_path, wav_file_name))

    def __track_is_tokenized(self, wav_file_name):
        return Path(join(self.tokenized_music_folder_path, wav_file_name[:-4] + '.npy')).exists()

    def __spectrogram_exists(self, wav_file_name):
        return Path(join(self.spectrograms_music_folder_path, wav_file_name[:-4] + '.npy')).exists()
