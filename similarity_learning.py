import sys
from os import listdir
from os.path import dirname
from os.path import join

import numpy
from numpy import shape
from sklearn.neighbors import NearestNeighbors


class NeighbourData:
    def __init__(self, origin, neighbour, distance):
        self.origin = origin
        self.neighbour = neighbour
        self.distance = distance


class SimilaritySegmentData:
    def __init__(self, index, similarity):
        self.index = index
        self.similarity = similarity


class SimilarityReport:
    def __init__(self, original_track_name, neighbour_data_list: list[NeighbourData],
                 segment_data_list: list[SimilaritySegmentData]):
        self.original_track_name = original_track_name
        self.neighbour_data_list = neighbour_data_list
        self.segment_data_list = segment_data_list


class SimilarityLearning:

    def __init__(self):
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')
        self.nn_model = NearestNeighbors()
        self.music_token_files = self.__filter_not_user_uploaded(listdir(self.tokenized_music_folder_path))[
                                 :100]  # TODO TEMPORARY FOR TESTING
        self.__train_model(self.music_token_files)

    def __train_model(self, music_token_files, segment: tuple[int, int] = None):
        samples = []
        for file in music_token_files:
            tokens = list(self.__load_mfcc_vector(file))
            tokens = self.__preserve_segment(tokens, segment)
            samples.append(tokens)

        print("Started learning...")
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
                  f'\t // with similarity of: {round(neighbour_data.distance * 100, 2)}%')

    def find_segment_similarities(self,
                                  uploaded_track_name,
                                  found_neighbours_data_list: list[NeighbourData],
                                  segments_amount=2):
        segment_data_list = []
        track_name = self.__trim_extension(uploaded_track_name)
        neighbour_token_files_list = list(
            map(lambda neighbour_data: neighbour_data.neighbour + '.npy', found_neighbours_data_list))

        mfcc_length = 128
        segment_length = round(mfcc_length / segments_amount)

        for i in range(segments_amount):
            segment = (segment_length * i, segment_length * (i + 1))
            self.__train_model(neighbour_token_files_list, segment)
            neighbour_data_list = self.__find_neighbours_for_track(track_name, neighbour_token_files_list, segment)
            print(f'### SEGMENT {i + 1}')
            self.__visualise_neighbours(neighbour_data_list)

            # segment_data_list.append(SimilaritySegmentData(i, ))

        return segment_data_list

    def __find_neighbours_for_track(self, track_name, token_files, segment: tuple[int, int] = None, neighbours_amount=10):
        neighbours = []
        mfcc = self.__preserve_segment(list(self.__load_mfcc_vector(track_name + '.npy')), segment)

        (distances_output, neighbours_output) = self.nn_model.kneighbors(
            [mfcc],
            n_neighbors=neighbours_amount,
            return_distance=True)

        for i in range(len(list(neighbours_output[0]))):
            distance = distances_output[0][i]
            neighbour = neighbours_output[0][i]

            neighbours.append(
                NeighbourData(origin=self.__trim_uploaded_postfix(track_name),
                              neighbour=self.__trim_extension(token_files[neighbour]),
                              distance=distance))

        self.__normalize_distances(neighbours)
        return neighbours

    @staticmethod
    def __normalize_distances(neighbour_data_list: list[NeighbourData]):
        max_distance = -1

        for neighbour_data in neighbour_data_list:
            max_distance = max(max_distance, neighbour_data.distance)
        for neighbour_data in neighbour_data_list:
            neighbour_data.distance = 1 - neighbour_data.distance / (max_distance * 1.1)

    def __load_mfcc_vector(self, file_name):
        return numpy.load(join(self.tokenized_music_folder_path, file_name))

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

    # SimilarityLearning().find_similar_tracks('$NOT - MEGAN.')
    model = SimilarityLearning()
    neighbours_list = model.find_similar_tracks('$uicideboy$ - WAR TIME ALL THE TIME')
    model.find_segment_similarities('$uicideboy$ - WAR TIME ALL THE TIME', neighbours_list)

# jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
