import os
import numpy as np
from PIL import Image
from IPython.display import display
import pandas as pd 

import librosa
import warnings

warnings.filterwarnings('ignore')

class NumpyCreator():
    
    '''연속된 이미지파일을 경로에서 가져와 60장의 concat된 Numpy array를 생성하는 클래스
    또한 그에 대응하는 y값과 text를 생성함
    
    Parameters
    ----------
    data : DataFrame
        영화제목 중복이 없는 DataFrame
        data에는 영화제목, 누적관객수, 제거, collected, file_num, text, multi_text 컬럼이 있어야함

    ratio : float
        상/하위 몇%로 자르는 기준
        
    
    '''
    def __init__(self, data, ratio:float):
        self.data = data
        self.ratio = ratio
        self.count = None
        self.best_data = None
        self.worst_data = None
        self.best_array = None
        self.worst_array = None
        self.X_text = None
        self.X_image = None
        self.y = None
        self.splitted = False
        
        
    def best_worst_splitter(self):
        #제거된 영화, 미수집 영화 제거
        self.filtered_data=self.data.query("(제거 == 0) & (collected == 'Y')").sort_values(by='누적관객수', ascending=False)
        
        #상위/하위 영화 추출개수 
        self.count = int(self.filtered_data.shape[0]*self.ratio)
        print(f"상위/하위 {self.ratio*100}% : 각각 {self.count:1.0f} 개")
        #file_num을 float으로 변환후 정렬 
        self.filtered_data['file_num'] = self.filtered_data['file_num'].astype(float)
        
        if 'new_genre' in self.filtered_data.columns:
            self.best_data = self.filtered_data[:self.count].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num', 'text','multi_text','new_genre']]
            self.worst_data = self.filtered_data[-self.count:].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num','text','multi_text','new_genre']]
        else:
            self.best_data = self.filtered_data[:self.count].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num', 'text','multi_text']]
            self.worst_data = self.filtered_data[-self.count:].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num','text','multi_text']]
        
        self.splitted = True
        print("Best 영화 샘플:")
        display(self.best_data.head(3))
        print()
        print("Worst 영화 샘플:")
        display(self.worst_data.head(3))

    def crop_and_resize_image(self, image_path):
        
        img = Image.open(image_path)
        
        width, height = img.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        img_cropped = img.crop((left, top, right, bottom))

        # 이미지 리사이즈
        img_resized = img_cropped.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img_resized)
        
        return img_array
    
    
    def create_numpy(self, file_lst, path):
        numpy_array = []
        for movie_folder in file_lst:   
            try:
                # print(f'movie folder number : {movie_folder}')
                movie_path = os.path.join(path, str(movie_folder))
                sequence = []
                
                image_count =0 
                for movie_images in sorted(os.listdir(movie_path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
                    image_count+=1
                    image_path = os.path.join(movie_path, movie_images)
                    img = self.crop_and_resize_image(image_path)
                    sequence.append(img)
                    
                    #순서대로 60장만 추출 
                    if image_count == 60: 
                        break
                numpy_array.append(np.array(sequence))
            except Exception as e:
                error_message = f'{movie_folder}번째 폴더없음 : {e}'
                print(error_message)
                raise ValueError(error_message) 

        return numpy_array
    
    def generate_text(self, multitext=False):
        #split 먼저 실행해서 self.best_data, self.worst_data 업데이트 
        if self.splitted is False: 
            self.best_worst_splitter()
        
        if multitext:
            best_text=self.best_data['multi_text'] 
            worst_text=self.worst_data['multi_text']
        else:
            best_text=self.best_data['text'] 
            worst_text=self.worst_data['text']

        self.X_text = np.array(pd.concat([best_text, worst_text], axis=0))
        
        #y
        self.y = np.zeros(len(best_text) + len(worst_text))
        self.y[:len(best_text)] = 1
        
        
        print("X_text shape:", self.X_text.shape)
        print("y shape:", self.y.shape)
        
        return self.X_text, self.y
    
    
    def generate_image(self, architecture='MobileNet'):
        #split 먼저 실행해서 self.best_data, self.worst_data 업데이트 
        if self.splitted is False: 
            self.best_worst_splitter()
            
        # 이미지 경로 
        best_path = f'/Users/eunseo/트레일러_영화흥행분석/Data/images/best'
        worst_path = f'/Users/eunseo/트레일러_영화흥행분석/Data/images/worst'
        best_file_lst = self.best_data['file_num'].astype(int).astype(str).tolist()
        worst_file_lst = self.worst_data['file_num'].astype(int).astype(str).tolist()
        try:
            print("best_array 생성시작...")
            self.best_array=self.create_numpy(best_file_lst, best_path)
            self.best_array=np.array(self.best_array)
            print("worst_array 생성시작...")
            self.worst_array=self.create_numpy(worst_file_lst, worst_path)
            self.worst_array = np.array(self.worst_array)
                
        except Exception as e:
            error_message = f"넘파이 생성 에러: {e}"
            print(error_message)
            raise ValueError(error_message)
        
        '''
        output shape example:

        best_train shape: (76, 100, 224, 224, 3)
        worst_train shape: (72, 100, 224, 224, 3)

        '''
        # print(f'best shape: {self.best_array.shape}')
        # print(f'worst shape: {self.worst_array.shape}')
        
        # 이미지 전처리
        if architecture == 'EfficientNet':
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif architecture == 'ResNet':
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
        elif architecture == 'VGGNet':
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif architecture == 'MobileNet':
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        else:
            raise ValueError("Invalid architecture choice. Choose from 'EfficientNet', 'ResNet', or 'VGGNet'.")
        
        # X_image 생성
        best_image = preprocess_input(self.best_array)
        worst_image = preprocess_input(self.worst_array)
        self.X_image = np.concatenate((best_image, worst_image), axis=0)

        
        # y생성
        self.y = np.zeros(self.X_image.shape[0])
        self.y[:self.best_array.shape[0]] = 1
        
        print("X_image shape:", self.X_image.shape)
        print("y shape:", self.y.shape)
            
        return  self.X_image, self.y
    
    
    def create_mfcc(self, label_lst, audio_path, max_length=7500):
        from sklearn import preprocessing
        
        mfcc_array = []
        
        file_lst=[]
        for label in label_lst:
            audiofile_list = os.listdir(audio_path)
            file_lst.extend([file for file in audiofile_list if file.split('_')[0] == label])
        
        # shapes = []
        for file in file_lst:
            file_path = os.path.join(audio_path, file)
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

            # MFCC 길이를 max_length에 맞게 자르거나 패딩을 추가
            mfccs = mfccs[:, :max_length] if mfccs.shape[1] >= max_length else np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
            mfccs = np.array(mfccs.T)
            mfcc_array.append(mfccs)
        
        mfcc_array = np.array(mfcc_array)        
        return mfcc_array
    
    
    def generate_audio(self, max_length=7500):       
        if self.splitted is False: 
            self.best_worst_splitter()
            
        #오디오 경로 
        best_path = f'/Users/eunseo/트레일러_영화흥행분석/Data/audios/best'
        worst_path = f'/Users/eunseo/트레일러_영화흥행분석/Data/audios/worst'
        best_file_lst = self.best_data['file_num'].astype(int).astype(str).tolist()
        worst_file_lst = self.worst_data['file_num'].astype(int).astype(str).tolist()
        
        try:
            print("best_array 생성시작...")
            self.best_audio=self.create_mfcc(best_file_lst, best_path, max_length=max_length)
            self.best_audio=np.array(self.best_audio)
            print("worst_array 생성시작...")
            self.worst_audio=self.create_mfcc(worst_file_lst, worst_path, max_length=max_length)
            self.worst_audio = np.array(self.worst_audio)
                
        except Exception as e:
            error_message = f"넘파이 생성 에러: {e}"
            print(error_message)
            raise ValueError(error_message)
        
        self.X_audio = np.concatenate((self.best_audio, self.worst_audio), axis=0)

        # y생성
        self.y = np.zeros(self.X_audio.shape[0])
        self.y[:self.best_audio.shape[0]] = 1
        
        print("X_audio shape:", self.X_audio.shape)
        print("y shape:", self.y.shape)
            
        return  self.X_audio, self.y
            
            
        
        
    def generate_data(self, multitext=False, architecture='MobileNet'):
        
        self.generate_text(multitext=multitext)
        self.generate_image(architecture=architecture)
        self.generate_audio()

        # print("X_text shape:", self.X_text.shape)
        # print("X_image shape:", self.X_image.shape)
        # print("X_audio shape:", self.X_audio.shape)
        # print("y shape:", self.y.shape)

        return self.X_text, self.X_image, self.X_audio, self.y



######################################구버전############################################
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from IPython.display import display
# import pandas as pd 

# class NumpyCreator():
#     '''연속된 이미지파일을 경로에서 가져와 60장의 concat된 Numpy array를 생성하는 클래스
#     또한 그에 대응하는 y값과 text를 생성함
    
#     Parameters
#     ----------
#     data : DataFrame
#         영화제목 중복이 없는 DataFrame
#         data에는 영화제목, 누적관객수, 제거, collected, file_num, text, multi_text 컬럼이 있어야함

#     ratio : float
#         상/하위 몇%로 자르는 기준
        
    
#     '''
#     def __init__(self, data, ratio:float):
#         self.data = data
#         self.ratio = ratio
#         self.count = None
#         self.best_data = None
#         self.worst_data = None
#         self.best_array = None
#         self.worst_array = None
#         self.X_text = None
#         self.X_image = None
#         self.y = None
        
        
#     def best_worst_splitter(self):
#         #제거된 영화, 미수집 영화 제거
#         self.filtered_data=self.data.query("(제거 == 0) & (collected == 'Y')").sort_values(by='누적관객수', ascending=False)
        
#         #상위/하위 영화 추출개수 
#         self.count = int(self.filtered_data.shape[0]*self.ratio)
#         print(f"상위/하위 {self.ratio*100}% : 각각 {self.count:1.0f} 개")
#         #file_num을 float으로 변환후 정렬 
#         self.filtered_data['file_num'] = self.filtered_data['file_num'].astype(float)
        
#         if 'new_genre' in self.filtered_data.columns:
#             self.best_data = self.filtered_data[:self.count].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num', 'text','multi_text','new_genre']]
#             self.worst_data = self.filtered_data[-self.count:].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num','text','multi_text','new_genre']]
#         else:
#             self.best_data = self.filtered_data[:self.count].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num', 'text','multi_text']]
#             self.worst_data = self.filtered_data[-self.count:].sort_values(by='file_num')[['영화제목','movie_code','누적관객수','file_num','text','multi_text']]
        
        
#         print("Best 영화 샘플:")
#         display(self.best_data.head(3))
#         print()
#         print("Worst 영화 샘플:")
#         display(self.worst_data.head(3))

#     def crop_and_resize_image(self, image_path):
        
#         img = Image.open(image_path)
        
#         width, height = img.size
#         new_size = min(width, height)
#         left = (width - new_size) / 2
#         top = (height - new_size) / 2
#         right = (width + new_size) / 2
#         bottom = (height + new_size) / 2
#         img_cropped = img.crop((left, top, right, bottom))

#         # 이미지 리사이즈
#         img_resized = img_cropped.resize((224, 224), Image.LANCZOS)
#         img_array = np.array(img_resized)
        
#         return img_array
    
    
#     def create_numpy(self, file_lst, path):
#         numpy_array = []
#         for movie_folder in file_lst:   
#             try:
#                 # print(f'movie folder number : {movie_folder}')
#                 movie_path = os.path.join(path, str(movie_folder))
#                 sequence = []
                
#                 image_count =0 
#                 for movie_images in sorted(os.listdir(movie_path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
#                     image_count+=1
#                     image_path = os.path.join(movie_path, movie_images)
#                     img = self.crop_and_resize_image(image_path)
#                     sequence.append(img)
                    
#                     #순서대로 60장만 추출 
#                     if image_count == 60: 
#                         break
#                 numpy_array.append(np.array(sequence))
#             except Exception as e:
#                 error_message = f'{movie_folder}번째 폴더없음 : {e}'
#                 print(error_message)
#                 raise ValueError(error_message) 

#         return numpy_array
    
#     def generate_text(self, multitext=False):
        
#         #split 먼저 실행해서 self.best_data, self.worst_data 업데이트 
#         self.best_worst_splitter()
        
#         if multitext:
#             best_text=self.best_data['multi_text'] 
#             worst_text=self.worst_data['multi_text']
#         else:
#             best_text=self.best_data['text'] 
#             worst_text=self.worst_data['text']

#         self.X_text = np.array(pd.concat([best_text, worst_text], axis=0))
        
#         #y
#         self.y = np.zeros(len(best_text) + len(worst_text))
#         self.y[:len(best_text)] = 1
        
#         return self.X_text, self.y
    
    
#     def generate_image(self, architecture='MobileNet'):
#         #split 먼저 실행해서 self.best_data, self.worst_data 업데이트 
#         self.best_worst_splitter()
#         # 이미지 경로 
#         best_path = f'/home/dlwnsfls/deepLearning_project/Data/images/best/'
#         worst_path = f'/home/dlwnsfls/deepLearning_project/Data/images/worst/'
#         best_file_lst = self.best_data['file_num'].astype(int).astype(str).tolist()
#         worst_file_lst = self.worst_data['file_num'].astype(int).astype(str).tolist()
#         try:
#             print("best_array 생성시작...")
#             self.best_array=self.create_numpy(best_file_lst, best_path)
#             self.best_array=np.array(self.best_array)
#             print("worst_array 생성시작...")
#             self.worst_array=self.create_numpy(worst_file_lst, worst_path)
#             self.worst_array = np.array(self.worst_array)
                
#         except Exception as e:
#             error_message = f"넘파이 생성 에러: {e}"
#             print(error_message)
#             raise ValueError(error_message)
        
#         '''
#         output shape example:

#         best_train shape: (76, 100, 224, 224, 3)
#         worst_train shape: (72, 100, 224, 224, 3)

#         '''
#         print(f'best shape: {self.best_array.shape}')
#         print(f'worst shape: {self.worst_array.shape}')
        
#         # 이미지 전처리
#         if architecture == 'EfficientNet':
#             from tensorflow.keras.applications.efficientnet import preprocess_input
#         elif architecture == 'ResNet':
#             from tensorflow.keras.applications.resnet_v2 import preprocess_input
#         elif architecture == 'VGGNet':
#             from tensorflow.keras.applications.vgg16 import preprocess_input
#         elif architecture == 'MobileNet':
#             from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#         else:
#             raise ValueError("Invalid architecture choice. Choose from 'EfficientNet', 'ResNet', or 'VGGNet'.")
        
#         # X_image 생성
#         best_image = preprocess_input(self.best_array)
#         worst_image = preprocess_input(self.worst_array)
#         self.X_image = np.concatenate((best_image, worst_image), axis=0)

        
#         # y생성
#         self.y = np.zeros(self.X_image.shape[0])
#         self.y[:self.best_array.shape[0]] = 1
            
#         return  self.X_image, self.y

#     def generate_data(self, multitext=False, architecture='MobileNet'):
        
#         self.generate_text(multitext=multitext)
#         self.generate_image(architecture=architecture)

#         print("X_text shape:", self.X_text.shape)
#         print("X_image shape:", self.X_image.shape)
#         print("y shape:", self.y.shape)

#         return self.X_text, self.X_image, self.y