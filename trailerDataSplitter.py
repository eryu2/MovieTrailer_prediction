from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class trailerDataSplitter():
    def __init__(self, y, X_text=None, X_image=None, X_audio=None, test_size=0.3, random_state=42, stratify=None):
        self.y = to_categorical(y)
        self.X_text = X_text
        self.X_image = X_image
        self.X_audio = X_audio
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify if stratify is not None else y
        
        
    def custom_train_test_split(self):
        
        if self.X_text is None:
            X_text_train = None
            X_text_test = None
        else:
            X_text_train, X_text_test, y_train, y_test = train_test_split(self.X_text, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify)
            print("X_text_train shape:", X_text_train.shape)
            print("X_text_test shape:", X_text_test.shape)
        if self.X_image is None:
            X_image_train = None
            X_image_test = None
        else:
            X_image_train, X_image_test, y_train, y_test = train_test_split(self.X_image, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify)
            print("X_image_train shape:", X_image_train.shape)
            print("X_image_test shape:", X_image_test.shape)
        if self.X_audio is None:
            X_audio_train = None
            X_audio_test = None
        else:
            X_audio_train, X_audio_test, y_train, y_test = train_test_split(self.X_audio, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.stratify)
            print("X_audio_train shape:", X_audio_train.shape)
            print("X_audio_test shape:", X_audio_test.shape)

        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)


        return X_text_train, X_text_test, X_image_train, X_image_test, X_audio_train, X_audio_test, y_train, y_test

        

