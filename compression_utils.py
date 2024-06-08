import numpy
import numpy as np


def __calculate_evenly_distributed_indexes(array_length, num_elements):
    if num_elements <= 0 or array_length <= 0:
        return []

    step = array_length / num_elements
    indexes = [int(i * step) for i in range(num_elements)]

    # Ensure indexes are within the bounds of the array length
    indexes = [min(index, array_length - 1) for index in indexes]

    return indexes


def compress_matrix(matrix, target_columns=3):
    columns_to_delete = len(matrix[0]) - target_columns

    indexes = __calculate_evenly_distributed_indexes(len(matrix[0]), columns_to_delete)
    for deleted_amount, index_to_delete in enumerate(indexes):
        current_index_to_delete = index_to_delete - deleted_amount
        first_list = [row[current_index_to_delete] for row in matrix]
        matrix = np.delete(matrix, current_index_to_delete, axis=1)
        next_element_index = current_index_to_delete % len(matrix[0])
        for i, first in enumerate(first_list):
            matrix[i][next_element_index] = (matrix[i][next_element_index] + first) / 2

    return matrix


def resize_matrix(matrix, target_length):
    if len(matrix[0]) < target_length:
        matrix = numpy.pad(array=matrix,
                           pad_width=((0, 0), (0, target_length - len(matrix[0]))),
                           mode='constant')
    return compress_matrix(matrix, target_columns=target_length)
