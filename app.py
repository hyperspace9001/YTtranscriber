import ffmpeg
import requests
import os
import pytube
import whisper
from pytube import YouTube
import subprocess
import yt_dlp as youtube_dl
import streamlit as st

class YouTubeTranscriber:
    def __init__(self, urls):
        self.urls = urls

    def download_video_mp3(self, youtube_url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloaded_video.mp3',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Get the absolute path of the downloaded file
        cwd = os.getcwd()
        audio_path = os.path.join(cwd, 'downloaded_video.mp3.mp3')

        return audio_path
    
    def transcribe(self, audio_path):
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path, fp16=False, language='English')
        return result["text"]

    def process_videos(self):
        yt_text = []
        for url in self.urls:
            audio_filename = self.download_video_mp3(url)
            transcription = self.transcribe(audio_filename)
            yt_text.append(transcription)
        return yt_text

def main():
    st.title('YouTube Video Transcriber')

    url = st.text_input('Enter YouTube URL:')
    if url:
        transcriber = YouTubeTranscriber([url])
        try:
            transcriptions = transcriber.process_videos()
            st.write(transcriptions)
        except Exception as e:
            st.write(f'Error: {e}')

if __name__ == "__main__":
    main()
