from google.cloud import speech_v1p1beta1 as speech
from tqdm import tqdm
import os
import pandas as pd



# 구글 클라우드 프로젝트 ID 및 서비스 계정 키 파일 경로 설정
project_id = 'aaa'
key_file = 'aaaa.json'


#폴더 경로 지정하기
folder_list =['best']

MOVIE_CODE = []
TXT_OUTPUT=[]

for i in folder_list:
    INPUT_PATH=f'Data/audios/{i}'
    
    #해당 폴더 안에 있는 파일 리스트 불러오기
    audio_list = sorted(list(os.listdir(INPUT_PATH)))

    # TXT_OUTPUT=[]
    for AUDIO in tqdm(audio_list):
        URI=f'gs://movie_audio_list/{i}/{AUDIO}'

        client = speech.SpeechClient.from_service_account_json(key_file)
        
        #config 만들기 
        config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
        audio_channel_count = 2,
        model = 'latest_long')
        
        #URI 정보 가져오기 
        audio = {"uri": URI}
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result()
        
        # 텍스트 한줄로 만들기 
        txt=""
        for result in response.results:
            txt+=result.alternatives[0].transcript+"."
        
        #리스트에 결과 저장
        TXT_OUTPUT.append(txt)
        movie_code = AUDIO.split('_')[2]
        MOVIE_CODE.append(movie_code)
        
#결과 저장하기
df = pd.DataFrame({'MOVIE_CODE':MOVIE_CODE,'TXT_OUTPUT':TXT_OUTPUT})
df.to_csv('./audio_text.csv',index=False)