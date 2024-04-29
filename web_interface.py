import os

import jsonpickle
from flask import Flask, url_for, redirect, flash, request

from similarity_learning import NeighbourData

UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'mp3'}

app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def open_index():
    return redirect(url_for('static', filename='index.html'), 302)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/find_similar_music", methods=["POST"])
def find_similar_music():
    query = request.args.get('file_name')

    # do similarity search sequence

    return jsonpickle.encode([NeighbourData("origin", "neighbour", 1.3)])


@app.route("/upload_music_file", methods=["POST"])
def upload_file():
    file = request.files['file']

    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    return redirect(url_for('static', filename='index.html'), 302)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
