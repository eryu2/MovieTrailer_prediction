from transformers import TFElectraModel
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, Dense, TimeDistributed, LSTM, Dropout, Concatenate, Add
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV2
from tensorflow.keras import layers


class ELECTRA(tf.keras.Model):
    def __init__(self):
        super(ELECTRA, self).__init__()

        self.electra_model = TFElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022", from_pt=True)
        for layer in self.electra_model.layers[:]:
            layer.trainable = True

        self.GAP = GlobalAveragePooling1D()
        self.flatten = Flatten()
        self.dense0 = Dense(64, activation='gelu')
        self.dense1 = Dense(8, activation='gelu')
        # self.dense2 = Dense(2, activation='softmax')

    def call(self, text_inputs):
        bert_outputs = self.electra_model(text_inputs)[0]
        output = self.GAP(bert_outputs)
        # output = self.flatten(bert_outputs)
        output = self.dense1(output)
        # output = self.dense2(output)
        return output



class ImageLSTMModel(tf.keras.Model): 
    def __init__(self, BACKBONE):
        super(ImageLSTMModel, self).__init__()

        if BACKBONE == 'EfficientNet':
            self.base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
        elif BACKBONE == 'ResNet':
            self.base_model = ResNet50V2(include_top=False, input_shape=(224, 224, 3))
        elif BACKBONE == 'VGGNet':
            self.base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
        elif BACKBONE == 'MobileNet':
            self.base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
        else:
            raise ValueError("Invalid BACKBONE choice.")

        for layer in self.base_model.layers[:]:
            layer.trainable = True

        self.time_distributed = TimeDistributed(self.base_model)
        self.time_distributed_GAP = TimeDistributed(GlobalAveragePooling2D())
        self.lstm = LSTM(16)
        self.dropout = Dropout(0.4)
        self.dense = Dense(8)

    def call(self, inputs):
        
        x = self.time_distributed(inputs)
        x = self.time_distributed_GAP(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

class AudioModel(tf.keras.Model):
    def __init__(self):
        super(AudioModel, self).__init__()

        self.conv1 = layers.Conv1D(8, 1, activation='relu')
        self.maxpool1 = layers.MaxPool1D(2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(8, activation='gelu')
        # self.dense2 = layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.flatten(x)
        output = self.dense1(x)

        return output
    
    
class MultiModalClassifier(tf.keras.Model):
    def __init__(self, BACKBONE):
        super(MultiModalClassifier, self).__init__()

        self.electra_model = ELECTRA()
        self.image_lstm_model = ImageLSTMModel(BACKBONE)
        self.concat = Concatenate()
        self.add = Add()
        self.dropout = Dropout(0.2)
        self.dense_1 = Dense(64, activation='gelu')
        self.dense_2 = Dense(16, activation='gelu')
        self.dense_3 = Dense(2, activation='softmax')

    def call(self, inputs):
        text_embeddings = inputs['text_embeddings']
        image_embeddings = inputs['image_embeddings']

        bert_output = self.electra_model(text_embeddings)
        image_lstm_output = self.image_lstm_model(image_embeddings)

        # combined_features = self.concat([bert_output, image_lstm_output])
        combined_features = self.add([bert_output, image_lstm_output])
        x = self.dense_2(combined_features)
        x = self.dropout(x)
        outputs = self.dense_3(x)
        return outputs



class MultiModalClassifier_audio(tf.keras.Model):
    def __init__(self, BACKBONE):
        super(MultiModalClassifier_audio, self).__init__()

        self.electra_model = ELECTRA()
        self.image_lstm_model = ImageLSTMModel(BACKBONE)
        self.AudioModel = AudioModel()
        self.concat = Concatenate()
        self.add = Add()
        self.dropout = Dropout(0.2)
        self.dense_1 = Dense(64, activation='gelu')
        self.dense_2 = Dense(16, activation='gelu')
        self.dense_3 = Dense(2, activation='softmax')

    def call(self, inputs):
        text_embeddings = inputs['text_embeddings']
        image_embeddings = inputs['image_embeddings']
        audio_embeddings = inputs['audio_embeddings']

        bert_output = self.electra_model(text_embeddings)
        image_lstm_output = self.image_lstm_model(image_embeddings)
        audio_output = self.AudioModel(audio_embeddings)

        # combined_features = self.concat([bert_output, image_lstm_output])
        combined_features = self.add([bert_output, image_lstm_output, audio_output])
        # combined_features = self.add([bert_output, image_lstm_output])
        # combined_features= self.concat([combined_features, audio_output])
        x = self.dense_1(combined_features)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        outputs = self.dense_3(x)
        return outputs





###############################이전버전############################################

# class ELECTRA(tf.keras.Model):
#     def __init__(self):
#         super(ELECTRA, self).__init__()

#         self.electra_model = TFElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022", from_pt=True)
#         for layer in self.electra_model.layers[:]:
#             layer.trainable = True

#         self.GAP = GlobalAveragePooling1D()
#         self.flatten = Flatten()
#         self.dense0 = Dense(64, activation='gelu')
#         self.dense1 = Dense(8, activation='gelu')
#         # self.dense2 = Dense(2, activation='softmax')

#     def call(self, text_inputs):
#         bert_outputs = self.electra_model(text_inputs)[0]
#         output = self.GAP(bert_outputs)
#         # output = self.flatten(bert_outputs)
#         output = self.dense1(output)
#         # output = self.dense2(output)
#         return output



# class ImageLSTMModel(tf.keras.Model): 
#     def __init__(self, BACKBONE):
#         super(ImageLSTMModel, self).__init__()

#         if BACKBONE == 'EfficientNet':
#             self.base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
#         elif BACKBONE == 'ResNet':
#             self.base_model = ResNet50V2(include_top=False, input_shape=(224, 224, 3))
#         elif BACKBONE == 'VGGNet':
#             self.base_model = VGG16(include_top=False, input_shape=(224, 224, 3))
#         elif BACKBONE == 'MobileNet':
#             self.base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
#         else:
#             raise ValueError("Invalid BACKBONE choice.")

#         for layer in self.base_model.layers[:]:
#             layer.trainable = True

#         self.time_distributed = TimeDistributed(self.base_model)
#         self.time_distributed_GAP = TimeDistributed(GlobalAveragePooling2D())
#         self.lstm = LSTM(16)
#         self.dropout = Dropout(0.4)
#         self.dense = Dense(8)

#     def call(self, inputs):
#         x = self.time_distributed(inputs)
#         x = self.time_distributed_GAP(x)
#         x = self.lstm(x)
#         x = self.dropout(x)
#         x = self.dense(x)
#         return x


# class MultiModalClassifier(tf.keras.Model):
#     def __init__(self, BACKBONE):
#         super(MultiModalClassifier, self).__init__()

#         self.electra_model = ELECTRA()
#         self.image_lstm_model = ImageLSTMModel(BACKBONE)
#         self.concat = Concatenate()
#         self.add = Add()
#         self.dropout = Dropout(0.2)
#         self.dense_1 = Dense(64, activation='gelu')
#         self.dense_2 = Dense(16, activation='gelu')
#         self.dense_3 = Dense(2, activation='softmax')

#     def call(self, inputs):
#         text_embeddings = inputs['text_embeddings']
#         image_embeddings = inputs['image_embeddings']

#         bert_output = self.electra_model(text_embeddings)
#         image_lstm_output = self.image_lstm_model(image_embeddings)

#         # combined_features = self.concat([bert_output, image_lstm_output])
#         combined_features = self.add([bert_output, image_lstm_output])
#         x = self.dense_2(combined_features)
#         x = self.dropout(x)
#         outputs = self.dense_3(x)
#         return outputs
