from os import listdir

from joblib import Parallel, delayed


class ConcurrencyUtils(object):

    @staticmethod
    def process_files_in_parallel(job, file_path, n_jobs=12):
        file_paths = listdir(file_path)
        Parallel(n_jobs=n_jobs)(delayed(job)(i, file_name, len(file_paths)) for i, file_name in enumerate(file_paths))
