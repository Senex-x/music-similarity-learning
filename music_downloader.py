import csv
import os
from os import path
from os.path import dirname

import spotipy
from savify import Savify
from savify.logger import Logger
from savify.types import Type, Format, Quality
from savify.utils import PathHolder
from spotipy.oauth2 import SpotifyClientCredentials


class SavifyService:

    def __init__(self):
        self.tracks_folder_path = path.join(dirname(__file__), 'data/music_mp3')
        self.savify = Savify(
            api_credentials=(os.getenv('SPOTIPY_CLIENT_ID'), os.getenv('SPOTIPY_CLIENT_SECRET')),
            quality=Quality.WORST,
            download_format=Format.MP3,
            path_holder=PathHolder(downloads_path=path.join(dirname(__file__), self.tracks_folder_path)),
            skip_cover_art=True,
            logger=Logger(log_location=path.join(dirname(__file__), 'data')),
            retry=2
        )

    def download(self, album_names):
        for album_name in album_names:
            self.savify.download(album_name, query_type=Type.ALBUM)


class MusicDownloader:

    def __init__(self):
        self.album_names_csv_path = path.join(dirname(__file__), 'data/registry/spotify_album_names.csv')
        self.album_names_path = path.join(dirname(__file__), 'data/registry/spotify_album_names.txt')
        self.album_ids_csv_path = path.join(dirname(__file__), 'data/registry/spotify_album_ids.csv')
        self.spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        self.savify = SavifyService()
        self.spotify.trace = False

    def download_tracks(self):
        album_names = self.__get_album_names()
        self.savify.download(album_names)

    def __get_album_names(self):
        album_names = []
        with open(self.album_names_path, mode='r', encoding='utf-8') as file:
            for album_name in file.readlines():
                album_names.append(album_name)
        return album_names
