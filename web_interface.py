import os
from os.path import join, dirname
import jsonpickle
import jsonpickle.ext.numpy
from flask import Flask, url_for, redirect, flash, request

from similarity_learning import NeighbourData, SimilarityLearning
from tokenizer import MusicTokenizer

UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
jsonpickle.ext.numpy.register_handlers()


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

        similar_tracks = SimilarityLearning().find_similar_tracks(track_file_name)

        os.remove(join(upload_folder_path, track_file_name))

        return jsonpickle.encode(similar_tracks)
    except:
        print('Error encountered')
        return jsonpickle.encode([])


@app.route("/upload_music_file", methods=["POST"])
def upload_file():
    file = request.files['file']

    if file and allowed_file(file.filename):
        track_extension = file.filename[file.filename.index('.'):]
        track_file_name = file.filename[:file.filename.index('.')] + "-user-uploaded" + track_extension
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], track_file_name))

    return redirect(url_for('static', filename='index.html'), 302)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
    # track_file_name = os.listdir(join(dirname(__file__), UPLOAD_FOLDER))[0]
    #
    # MusicTokenizer().tokenize_uploaded_track()
    #
    # similar_tracks = SimilarityLearning().find_similar_tracks(track_file_name)

