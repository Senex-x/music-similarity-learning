from os import listdir
from os.path import dirname
from os.path import join

import numpy
from sklearn.neighbors import NearestNeighbors


class NeighbourData:

    def __init__(self, origin, neighbour, distance):
        self.origin = origin
        self.neighbour = neighbour
        self.distance = distance


class SimilarityLearning:

    def __init__(self):
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')
        self.nn_model = NearestNeighbors()

    def learn(self):
        samples = []
        music_token_files = listdir(self.tokenized_music_folder_path)[:1000]
        for file in music_token_files:
            samples.append(self.__load_mfcc_vector(file).tolist())

        self.nn_model.fit(samples)

        neighbour_data_list = self.__find_neighbours_for_track('A Day To Remember - Homesick', music_token_files)

        print(f'Track: {neighbour_data_list[0].origin} is most similar to:')
        for i, neighbour_data in enumerate(neighbour_data_list):
            print(f'{i + 1}: {neighbour_data.neighbour} \t // with similarity of: {round(neighbour_data.distance * 100, 2)}%')

        # self.__find_nearest_neighbour_for_each_track(nn_model, music_token_files)

    def __find_neighbours_for_track(self, track_name, token_files, neighbours_amount=10):
        neighbours = []
        (distances_output, neighbours_output) = self.nn_model.kneighbors(
            [self.__load_mfcc_vector(track_name + '.npy').tolist()],
            n_neighbors=neighbours_amount + 1,
            return_distance=True)

        for i in range(len(list(neighbours_output[0])))[1:]:
            distance = distances_output[0][i]
            neighbour = neighbours_output[0][i]

            neighbours.append(
                NeighbourData(origin=track_name,
                              neighbour=self.__trim_extension(token_files[neighbour]),
                              distance=distance))

        self.__normalize_distances(neighbours)
        return neighbours

    def __find_nearest_neighbour_for_each_track(self, token_files):
        neighbour_data_list = self.__associate_tracks_with_nearest_neighbour(token_files)
        self.__normalize_distances(neighbour_data_list)

        for neighbour_data in neighbour_data_list:
            print(f'Track: {neighbour_data.origin} \t is most similar to: \t {neighbour_data.neighbour} '
                  f'\t with similarity of: \t {round(neighbour_data.distance * 100, 2)}%')

    @staticmethod
    def __normalize_distances(neighbour_data_list):
        max_distance = -1

        for neighbour_data in neighbour_data_list:
            max_distance = max(max_distance, neighbour_data.distance)
        for neighbour_data in neighbour_data_list:
            neighbour_data.distance = 1 - neighbour_data.distance / max_distance

    # Track: Bad Bunny - La Corriente.npy 	 is most similar to: 	 Avicii - Talk To Myself.npy with similarity of: 	 90.66%
    # Track: Baby Keem - trademark usa.npy 	 is most similar to: 	 A$AP Rocky - Holy Ghost (feat. Joe Fox).npy with similarity of: 	 91.25%
    # Track: Ariana Grande - no tears left to cry.npy 	 is most similar to: 	 Ariana Grande - Hands On Me.npy with similarity of: 	 90.96%
    # Track: Arctic Monkeys - From The Ritz To The Rubble.npy 	 is most similar to: 	 Arctic Monkeys - The View From The Afternoon.npy with similarity of: 	 93.48%
    # Track: Alesso - If It Wasn't For You.npy 	 is most similar to: 	 Bastille - Things We Lost In The Fire.npy with similarity of: 	 91.22%
    # Track: Akon - I Can't Wait.npy 	 is most similar to: 	 Baby Keem - family ties (with Kendrick Lamar).npy with similarity of: 	 91.11%
    # Track: A$AP Rocky - Jodye.npy 	 is most similar to: 	 A$AP Rocky - Lord Pretty Flacko Jodye 2 (LPFJ2).npy with similarity of: 	 100.0%
    # Track: A$AP Rocky - Ghetto Symphony (feat. Gunplay & A$AP Ferg).npy 	 is most similar to: 	 Alan Walker - Alone, Pt. II.npy with similarity of: 	 92.71%

    def __associate_tracks_with_nearest_neighbour(self, token_files):
        neighbours = []
        for file in token_files:
            nearest_neighbours = self.nn_model.kneighbors([self.__load_mfcc_vector(file).tolist()],
                                                          n_neighbors=2,
                                                          return_distance=True)
            nearest_neighbour = NeighbourData(origin=self.__trim_extension(file),
                                              neighbour=self.__trim_extension(token_files[nearest_neighbours[1][0][1]]),
                                              distance=nearest_neighbours[0][0][1])
            neighbours.append(nearest_neighbour)
        return neighbours

    def __load_mfcc_vector(self, file_name):
        return numpy.load(join(self.tokenized_music_folder_path, file_name))

    @staticmethod
    def __trim_extension(file_name):
        return file_name[:file_name.index('.')]


if __name__ == '__main__':
    print("Running")

    SimilarityLearning().learn()
