from os import listdir
from os.path import dirname
from os.path import join
from pathlib import Path

from joblib import Parallel, delayed
from pydub import AudioSegment
from pylab import *


class MusicTokenizer:

    def __init__(self):
        self.music_folder_path = join(dirname(__file__), 'data/music_mp3')
        self.converted_music_folder_path = join(dirname(__file__), 'data/music_wav')
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')

    def tokenize(self):
        self.__convert_mp3_files_to_wav()

    def __convert_mp3_files_to_wav(self):
        mp3_file_names = listdir(self.music_folder_path)
        for i, mp3_file_name in enumerate(mp3_file_names):
            if not self.__track_is_converted(mp3_file_name):
                sound = AudioSegment.from_mp3(join(self.music_folder_path, mp3_file_name))
                sound.export(out_f=join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav'),
                             format="wav",
                             codec="pcm_u8",
                             parameters=["-ac", "1", "-ar", "4000"])
            print(f'Converted {i + 1} out of {len(mp3_file_names)} .mp3 files '
                  f'({round((i + 1) / len(mp3_file_names) * 100, 2)}%)')

    def __track_is_converted(self, mp3_file_name):
        return Path(join(self.converted_music_folder_path, mp3_file_name[:-4] + '.wav')).exists()

    def __track_is_tokenized(self, wav_file_name):
        return Path(join(self.tokenized_music_folder_path, wav_file_name[:-4] + '.npy')).exists()

    @staticmethod
    def __execute_in_parallel(job, file_path):
        file_paths = listdir(file_path)
        Parallel(n_jobs=12)(delayed(job)(i, file_name, len(file_paths)) for i, file_name in enumerate(file_paths))


if __name__ == '__main__':
    MusicTokenizer().tokenize()