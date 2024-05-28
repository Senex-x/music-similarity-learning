import numpy
import numpy as np


def calculate_evenly_distributed_indexes(array_length, num_elements):
    if num_elements <= 0 or array_length <= 0:
        return []

    step = array_length / num_elements
    indexes = [int(i * step) for i in range(num_elements)]

    # Ensure indexes are within the bounds of the array length
    indexes = [min(index, array_length - 1) for index in indexes]

    return indexes


def compress_matrix(matrix, target_columns=3):
    columns_to_delete = len(matrix[0]) - target_columns

    indexes = calculate_evenly_distributed_indexes(len(matrix[0]), columns_to_delete)
    for deleted_amount, index_to_delete in enumerate(indexes):
        current_index_to_delete = index_to_delete - deleted_amount
        # print(f'index to delete {current_index_to_delete}, matrix {matrix[i]}')
        first = matrix[0][current_index_to_delete]
        matrix = np.delete(matrix, current_index_to_delete, axis=1)
        next_element_index = current_index_to_delete % len(matrix[0])
        matrix[0][next_element_index] = (matrix[0][next_element_index] + first) / 2

    return matrix


if __name__ == '__main__':
    # MusicDownloader().download_tracks()
    matrix = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    # matrix = numpy.pad(matrix, ((0, 0), (0, 5)), mode='constant')
    print(matrix)
    print(compress_matrix(matrix))
