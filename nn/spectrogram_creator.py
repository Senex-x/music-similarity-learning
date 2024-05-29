from os import listdir, remove
from os.path import dirname
from os.path import join
from pathlib import Path

import librosa
import librosa.display
import numpy
from pydub import AudioSegment
from pylab import *

import compression_utils
from concurrency_utils import ConcurrencyUtils
import matplotlib.pyplot as plt
import numpy as np


class SpectrogramCreator:

    def __init__(self):
        self.converted_music_folder_path = join(dirname(__file__), '../data/music_wav')
        self.spectrograms_music_folder_path = join(dirname(__file__), '../data/music_spectrograms')

    def create_spectrogram_from_wav_file(self, wav_file_name):
        if not self.__spectrogram_exists(wav_file_name):
            try:
                x, sample_rate = librosa.load(join(self.converted_music_folder_path, wav_file_name),
                                              mono=True,
                                              res_type='kaiser_fast')
                # mfcc_matrix = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=128)
                # output: coefficients to time frames

                S = librosa.feature.melspectrogram(y=x, sr=sample_rate, n_mels=128, fmax=2000)  # 2k is ok perhaps
                mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=30)

                fig, ax = plt.subplots(nrows=5, sharex=False)
                plt.subplots_adjust(hspace=0.8)
                img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                               x_axis='time', y_axis='mel', fmax=8000,
                                               ax=ax[0])
                fig.colorbar(img, ax=[ax[0]])
                # fig.set_figwidth(8)
                fig.set_figheight(10)
                ax[0].set(title='Mel spectrogram')
                ax[0].label_outer()

                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
                fig.colorbar(img, ax=[ax[1]])
                ax[1].set(title='MFCC')

                mfccs = compression_utils.compress_matrix(mfccs, 500)
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[2])
                fig.colorbar(img, ax=[ax[2]])
                ax[2].set(title='Compressed MFCC')

                mfccs = self.apply_distortion(mfccs)
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[3])
                fig.colorbar(img, ax=[ax[3]])
                ax[3].set(title='Distorted MFCC')

                mfccs = self.make_negative_matrix(mfccs)
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[4])
                fig.colorbar(img, ax=[ax[4]])
                ax[4].set(title='Negative MFCC')

                # plt.subplots_adjust(wspace=1)

                plt.show()

                # print(mfccs)
                # print(shape(mfccs))

                # numpy.save(join(self.spectrograms_music_folder_path, wav_file_name[:-4]), mfccs)
            except:
                print(f"Error while tokenizing {wav_file_name}")
                raise
                # remove(join(self.converted_music_folder_path, wav_file_name))

    def __spectrogram_exists(self, wav_file_name):
        return Path(join(self.spectrograms_music_folder_path, wav_file_name[:-4] + '.npy')).exists()

    def apply_distortion(self, matrix):
        noise = np.random.normal(0, 30, shape(matrix))
        print(shape(noise))
        print(f'noise sample {noise[:5]}')

        return numpy.add(matrix, noise)

    def make_negative_matrix(self, matrix):
        noise = np.random.normal(0, 300, shape(matrix))
        matrix = np.add(matrix, noise)
        for row in matrix:
            for j in range(len(row)):
                if row[j] >= 0:
                    row[j] = row[j] % 200
                else:
                    row[j] = -(-row[j] % 500)
        return matrix


if __name__ == '__main__':

    SpectrogramCreator().create_spectrogram_from_wav_file('Piano Rockstar - Tides - Piano Version.wav')
