import os

import spotipy
from savify.logger import Logger
from spotipy.oauth2 import SpotifyClientCredentials
from os import path
from os.path import dirname
import csv
from savify import Savify
from savify.types import Type, Format, Quality
from savify.utils import PathHolder


class SavifyService:

    def __init__(self):
        self.tracks_folder_path = path.join(dirname(__file__), 'data\\files\\music')
        self.savify = Savify(
            api_credentials=(os.getenv('SPOTIPY_CLIENT_ID'), os.getenv('SPOTIPY_CLIENT_SECRET')),
            quality=Quality.WORST,
            download_format=Format.MP3,
            path_holder=PathHolder(downloads_path=path.join(dirname(__file__), 'data\\files\\music')),
            skip_cover_art=True,
            logger=Logger(log_location=path.join(dirname(__file__), 'data\\files')))

    def download(self, album_names):
        for album_name in album_names:
            self.savify.download(album_name, query_type=Type.ALBUM)


class MusicDownloader:

    def __init__(self):
        self.album_names_csv_path = path.join(dirname(__file__), 'data\\files\\spotify_album_names.csv')
        self.album_ids_csv_path = path.join(dirname(__file__), 'data\\files\\spotify_album_ids.csv')
        self.spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        self.savify = SavifyService()
        self.spotify.trace = False

    def download_tracks(self):
        album_names = self.__get_album_names()
        self.savify.download(album_names)

    def __get_album_names(self):
        album_names = []
        with open(self.album_names_csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                album_names.append(row[3])
        return album_names[1:]


if __name__ == '__main__':
    MusicDownloader().download_tracks()
