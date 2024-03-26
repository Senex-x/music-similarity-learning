from sklearn.neighbors import NearestNeighbors
import numpy
from os.path import join
from os import listdir
from os.path import dirname


class SimilarityLearning:

    def __init__(self):
        self.tokenized_music_folder_path = join(dirname(__file__), 'data/music_tokens')

    def learn(self):
        neigh = NearestNeighbors(n_neighbors=5, radius=1)
        samples = []

        music_token_files = listdir(self.tokenized_music_folder_path)
        for file in music_token_files:
            samples.append(self.__load_mfcc_vector(file).tolist())

        neigh.fit(samples)
        neighbours = neigh.kneighbors([self.__load_mfcc_vector("2Pac - Ain't Hard 2 Find (ft. B-Legit, C-BO, E-40, Richie Rich).npy").tolist()], 2, return_distance=False)
        print(neighbours)

        nearest_neighbour = music_token_files[neighbours[0][1]] # Baby Keem - cocoa (with Don Toliver)
        print(nearest_neighbour)

    def __load_mfcc_vector(self, file_name):
        return numpy.load(join(self.tokenized_music_folder_path, file_name))


if __name__ == '__main__':
    print("Running")

    SimilarityLearning().learn()

