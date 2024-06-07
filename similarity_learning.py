import math
import string
import sys
from os import listdir
from os.path import dirname
from os.path import join
import librosa
import numpy
import numpy as np
from mutagen.wave import WAVE
from numpy import shape
from sklearn.neighbors import NearestNeighbors as SimSLRModel
import mutagen
import time


class NeighbourData:
    def __init__(self, origin, neighbour, similarity, origin_total_duration, total_duration):
        self.origin = origin
        self.neighbour = neighbour
        self.similarity = similarity
        self.origin_total_duration = origin_total_duration
        self.total_duration = total_duration


class SimilaritySegmentData:
    def __init__(self, similarity, duration_start, duration_end):
        self.similarity = similarity
        self.duration_start = duration_start
        self.duration_end = duration_end


class SimilarityReport:
    def __init__(self,
                 original_track_name,
                 original_track_total_duration,
                 neighbour_data_list: list[NeighbourData],
                 segment_data_list: dict[list[SimilaritySegmentData]]):
        self.original_track_name = original_track_name
        self.original_track_total_duration = original_track_total_duration
        self.neighbour_data_list = neighbour_data_list
        self.segment_data_list = segment_data_list


class SimilarityLearning:

    def __init__(self):
        self.track_segments_amount = 8
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')
        self.wav_music_folder_path = join(dirname(__file__), 'data/music_wav')
        self.nn_model = SimSLRModel()
        self.nn_model_segmented = SimSLRModel()
        self.music_token_files = self.__filter_not_user_uploaded(listdir(self.tokenized_music_folder_path))
        self.__train_model(self.music_token_files)

    def __train_model(self, music_token_files, segment: tuple[int, int] = None):
        samples = []
        for file in music_token_files:
            tokens = list(self.__load_mfcc_vector(file))
            tokens = self.__preserve_segment(tokens, segment)
            samples.append(tokens)

        print("Started learning...")
        if segment:
            self.nn_model_segmented.fit(samples)
        else:
            self.nn_model.fit(samples)
        print("Finished learning")

    @staticmethod
    def __preserve_segment(vector: list, segment: tuple[int, int] = None):
        if segment:
            return [0] * segment[0] + vector[segment[0]:segment[1]] + [0] * (len(vector) - segment[1])
        else:
            return vector

    def find_similar_tracks(self, track_file_name, neighbour_amount=10):
        neighbour_data_list = self.__find_neighbours_for_track(self.__trim_extension(track_file_name),
                                                               self.music_token_files,
                                                               None,
                                                               neighbour_amount)
        self.__visualise_neighbours(neighbour_data_list)
        # self.__find_nearest_neighbour_for_each_track(nn_model, music_token_files)
        return neighbour_data_list

    @staticmethod
    def __visualise_neighbours(neighbour_data_list: list[NeighbourData]):
        print(f'Track: {neighbour_data_list[0].origin} is most similar to:')
        for i, neighbour_data in enumerate(neighbour_data_list):
            print(f'{i + 1}: {neighbour_data.neighbour}'
                  f'\t // with similarity of: {neighbour_data.similarity}%')

    def find_segment_similarities(self,
                                  uploaded_track_name,
                                  found_neighbours_data_list: list[NeighbourData]):
        track_name = self.__trim_extension(uploaded_track_name)
        neighbour_token_files_list = list(
            map(lambda neighbour_data: neighbour_data.neighbour + '.npy', found_neighbours_data_list))

        mfcc_length = 128
        segment_length = round(mfcc_length / self.track_segments_amount)
        segment_reports = {}

        for i in range(self.track_segments_amount):
            print(f'### SEGMENT {i + 1}')
            segment = (segment_length * i, segment_length * (i + 1))
            self.__train_model(neighbour_token_files_list, segment)
            neighbour_data_list = self.__find_neighbours_for_track(track_name, neighbour_token_files_list, segment)
            self.__visualise_neighbours(neighbour_data_list)

            for neighbour_data in neighbour_data_list:
                start, end = self.__get_track_duration_formatted(neighbour_data.neighbour, segment_index=i)
                current_value = segment_reports.get(neighbour_data.neighbour)
                new_value = SimilaritySegmentData(neighbour_data.similarity, duration_start=start, duration_end=end)
                if current_value:
                    current_value.append(new_value)
                else:
                    segment_reports[neighbour_data.neighbour] = [new_value]

        return segment_reports

    def __find_neighbours_for_track(self, track_name,
                                    token_files,
                                    segment: tuple[int, int] = None,
                                    neighbours_amount=10):
        neighbours = []
        mfcc = self.__preserve_segment(list(self.__load_mfcc_vector(track_name + '.npy')), segment)

        if segment:
            model = self.nn_model_segmented
        else:
            model = self.nn_model

        (distances_output, neighbours_output) = model.kneighbors(
            [mfcc],
            n_neighbors=neighbours_amount,
            return_distance=True)

        for i in range(len(list(neighbours_output[0]))):
            distance = distances_output[0][i]
            neighbour = neighbours_output[0][i]
            _, origin_duration = self.__get_track_duration_formatted(track_name)
            _, duration = self.__get_track_duration_formatted(self.__trim_extension(token_files[neighbour]))

            neighbours.append(
                NeighbourData(origin=self.__trim_uploaded_postfix(track_name),
                              neighbour=self.__trim_extension(token_files[neighbour]),
                              similarity=distance,
                              origin_total_duration=origin_duration,
                              total_duration=duration))

        self.__normalize_distances(neighbours, segment)
        return neighbours

    def __normalize_distances(self, neighbour_data_list: list[NeighbourData], segment: tuple[int, int] = None):
        max_distance = 0

        for neighbour_data in neighbour_data_list:
            if segment:
                neighbour_data.similarity = neighbour_data.similarity * self.track_segments_amount
            max_distance = max(max_distance, neighbour_data.similarity)

        for i, neighbour_data in enumerate(neighbour_data_list):
            similarity_score = 1 - neighbour_data.similarity / (max_distance * 1.1)
            if segment and i != 0:
                similarity_score /= 3
            neighbour_data.similarity = round(similarity_score * 100, 2)

    def __load_mfcc_vector(self, file_name):
        return numpy.load(join(self.tokenized_music_folder_path, file_name))

    def __get_track_duration_formatted(self, track_name, segment_index=None) -> tuple[string, string]:
        track_name = join(self.wav_music_folder_path, track_name + '.wav')
        start_duration = 0
        end_duration = librosa.get_duration(path=track_name)
        if segment_index or segment_index == 0:
            start_duration = segment_index * end_duration / self.track_segments_amount
            end_duration = start_duration + end_duration / self.track_segments_amount
        return time.strftime('%M:%S', time.gmtime(start_duration)), time.strftime('%M:%S', time.gmtime(end_duration))

    @staticmethod
    def __trim_extension(file_name):
        try:
            return file_name[:file_name.rindex('.')]
        except:
            return file_name

    @staticmethod
    def __trim_uploaded_postfix(string):
        return string.replace("-user-uploaded", '')

    @staticmethod
    def __filter_not_user_uploaded(file_names):
        return list(filter(lambda file_name: "-user-uploaded" not in file_name, file_names))

    # def __associate_tracks_with_nearest_neighbour(self, token_files):


#     neighbours = []
#     for file in token_files:
#         nearest_neighbours = self.nn_model.kneighbors([self.__load_mfcc_vector(file).tolist()],
#                                                       n_neighbors=2,
#                                                       return_distance=True)
#         nearest_neighbour = NeighbourData(origin=self.__trim_extension(file),
#                                           neighbour=self.__trim_extension(token_files[nearest_neighbours[1][0][1]]),
#                                           distance=nearest_neighbours[0][0][1])
#         neighbours.append(nearest_neighbour)
#     return neighbours

# def __find_nearest_neighbour_for_each_track(self, token_files):
#     neighbour_data_list = self.__associate_tracks_with_nearest_neighbour(token_files)
#     self.__normalize_distances(neighbour_data_list)
#
#     for neighbour_data in neighbour_data_list:
#         print(f'Track: {neighbour_data.origin} \t is most similar to: \t {neighbour_data.neighbour} '
#               f'\t with similarity of: \t {round(neighbour_data.distance * 100, 2)}%')


if __name__ == '__main__':
    print("Running")

    # example
    # model = SimilarityLearning()
    # neighbours_list = model.find_similar_tracks('$uicideboy$ - WAR TIME ALL THE TIME')
    # segment_similarity = model.find_segment_similarities('$uicideboy$ - WAR TIME ALL THE TIME', neighbours_list)
    # SimilarityReport(
    #     '$uicideboy$ - WAR TIME ALL THE TIME',
    #     neighbours_list,
    #     segment_similarity
    # )

# jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
