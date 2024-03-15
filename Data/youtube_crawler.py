from pytube import YouTube

def download_video(video_url, output_path, output_filename):
    yt = YouTube(video_url)
    stream = yt.streams.filter(res='720p', file_extension='mp4').first()
    stream.download(output_path=output_path, filename=output_filename)
