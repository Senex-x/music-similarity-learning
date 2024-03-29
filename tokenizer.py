from os.path import dirname
from os.path import join
from pathlib import Path

import librosa
import librosa.display
import numpy
from pydub import AudioSegment
from pylab import *

from concurrency_utils import ConcurrencyUtils


class MusicTokenizer:

    def __init__(self):
        self.music_folder_path = join(dirname(__file__), 'data/music_mp3')
        self.converted_music_folder_path = join(dirname(__file__), 'data/music_wav')
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')

    def tokenize(self):
        self.__convert_mp3_files_to_wav()
        self.__extract_feature_vectors()

    def __convert_mp3_files_to_wav(self):
        def job(i, mp3_file_name, file_count):
            if not self.__track_is_converted(mp3_file_name):
                sound = AudioSegment.from_mp3(join(self.music_folder_path, mp3_file_name))
                sound.export(out_f=join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav'),
                             format="wav",
                             codec="pcm_u8",
                             parameters=["-ac", "1", "-ar", "4000"])
            print(f'Converted {i + 1} out of {file_count} .mp3 files '
                  f'({round((i + 1) / file_count * 100, 2)}%)')

        ConcurrencyUtils.process_files_in_parallel(job, self.music_folder_path)

    def __track_is_converted(self, mp3_file_name):
        return Path(join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav')).exists()

    def __extract_feature_vectors(self):
        def job(i, wav_file_name, file_count):
            if not self.__track_is_tokenized(wav_file_name):
                # load the audio file
                x, sample_rate = librosa.load(join(self.converted_music_folder_path, wav_file_name),
                                              res_type='kaiser_fast')
                # extract features from the audio
                mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)

                numpy.save(join(self.tokenized_music_folder_path, wav_file_name[:-4]), mfcc)

            print(f'Tokenized {i + 1} out of {file_count} .wav files '
                  f'({round((i + 1) / file_count * 100, 2)}%)')

        ConcurrencyUtils.process_files_in_parallel(job, self.converted_music_folder_path)

    def __track_is_tokenized(self, wav_file_name):
        return Path(join(self.tokenized_music_folder_path, wav_file_name[:-4] + '.npy')).exists()


if __name__ == '__main__':
    MusicTokenizer().tokenize()
