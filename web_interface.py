import os
from os.path import join, dirname
import jsonpickle
import jsonpickle.ext.numpy
from flask import Flask, url_for, redirect, flash, request

from similarity_learning import NeighbourData, SimilarityLearning, SimilarityReport
from tokenizer import MusicTokenizer

UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
jsonpickle.ext.numpy.register_handlers()

similarity_learning_model = SimilarityLearning()


@app.route('/', methods=['GET'])
def open_index():
    return redirect(url_for('static', filename='index.html'), 302)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/find_similar_music", methods=["POST"])
def find_similar_music():
    upload_folder_path = join(dirname(__file__), UPLOAD_FOLDER)
    try:
        track_file_name = os.listdir(upload_folder_path)[0]

        MusicTokenizer().tokenize_uploaded_track()

        similar_tracks = similarity_learning_model.find_similar_tracks(track_file_name)
        segment_similarity = similarity_learning_model.find_segment_similarities(track_file_name, similar_tracks)

        __remove_uploaded_files(upload_folder_path, track_file_name)

        return jsonpickle.encode(
            SimilarityReport(
                similar_tracks[0].origin,
                similar_tracks[0].origin_total_duration,
                similar_tracks,
                segment_similarity
            )
        )
    except Exception as e:
        print(f'Error encountered: {e}')
        return jsonpickle.encode([])


@app.route("/upload_music_file", methods=["POST"])
def upload_file():
    file = request.files['file']

    if file and allowed_file(file.filename):
        track_extension = file.filename[file.filename.index('.'):]
        track_file_name = file.filename[:file.filename.index('.')] + "-user-uploaded" + track_extension
        __clear_folder(app.config['UPLOAD_FOLDER'])
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], track_file_name))

    return redirect(url_for('static', filename='index.html'), 302)


def __remove_uploaded_files(upload_folder_path, uploaded_file_name):
    try:
        os.remove(join(upload_folder_path, uploaded_file_name))
        os.remove(join(dirname(__file__), 'data/music_wav', uploaded_file_name[:-4] + '.wav'))
        os.remove(join(dirname(__file__), 'data/music_tokens', uploaded_file_name[:-4] + '.npy'))
    except Exception as e:
        print('Failed to remove uploaded files')


def __clear_folder(folder_path):
    for file_name in os.listdir(folder_path):
        try:
            os.remove(join(folder_path, file_name))
        except Exception as e:
            print(f'Failed to delete file with name={file_name}')


if __name__ == '__main__':
    app.run(port=8080, debug=True)
    # track_file_name = os.listdir(join(dirname(__file__), UPLOAD_FOLDER))[0]
    #
    # MusicTokenizer().tokenize_uploaded_track()
    #
    # similar_tracks = SimilarityLearning().find_similar_tracks(track_file_name)

