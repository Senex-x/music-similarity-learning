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

        for pair in self.__associate_track_with_nearest_neighbour(neigh, music_token_files):
            print(f'Track: {pair[0]} \t is most similar to: \t {pair[1]}')

    # Track: Ariana Grande - everytime.npy 	 is most similar to: 	 Akon - Mama Africa.npy
    # Track: alt - J - Breezeblocks.npy is most similar to: Bastille - Icarus.npy
    # Track: Akon - I'm So Paid.npy 	 is most similar to: 	 Alan Walker - Alone.npy
    # Track: A$AP Rocky - Max B(feat.Joe Fox).npy is most similar to: Bad Bunny - Efecto.npy
    # Track: 21 Savage - Intro.npy 	 is most similar to: 	 Adele - Rolling in the Deep.npy
    # Track: Alec Benjamin - Steve.npy 	 is most similar to: 	 Ali Gatie - Million Miles Apart.npy
    # Track: Ali Gatie - Somebody Else.npy 	 is most similar to: 	 Bad Bunny - Otro Atardecer.npy

    def __associate_track_with_nearest_neighbour(self, nn_model, token_files):
        neighbours = []

        for file in token_files:
            nearest_neighbours = nn_model.kneighbors([self.__load_mfcc_vector(file).tolist()], 2, return_distance=False)
            nearest_neighbour = token_files[nearest_neighbours[0][1]]
            neighbours.append((file, nearest_neighbour))

        return neighbours

    def __load_mfcc_vector(self, file_name):
        return numpy.load(join(self.tokenized_music_folder_path, file_name))


if __name__ == '__main__':
    print("Running")

    SimilarityLearning().learn()

