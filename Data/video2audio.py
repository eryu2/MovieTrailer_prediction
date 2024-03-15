from moviepy.editor import VideoFileClip
import os


#폴더 경로 지정, 폴더 안에 있는 파일 리스트 불러오기
path='Data/videos'
videos_path = os.listdir(path)

OUTPUT_PATH = 'Data/audios/'
# 오디오 추출
for video in videos_path:
    video_clip = VideoFileClip(video)
    audio_clip = video_clip.audio

    # 추출된 오디오를 WAV 파일로 저장
    NAME=video.split('.')[0]
    output_fname =f'{OUTPUT_PATH}{NAME}.wav'
    audio_clip.write_audiofile(output_fname)

    # 메모리 해제
    audio_clip.close()