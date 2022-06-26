from flask import Flask, render_template, request, url_for, make_response, jsonify, redirect
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_wavelets.Layers.DWT as wavelet
from keras.preprocessing.image import ImageDataGenerator
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/CNN', methods=['GET', 'POST'])
def login():
   message = None
   if request.method == 'POST':
        datafromjs = request.form['data']
        if(datafromjs=="1"):
            result = "is 1"
            resp = make_response('{"response": "sdfsdfsdfsdfds"}')
            resp.headers['Content-Type'] = "application/json"
        else:
            result = "is not 1"
            resp = make_response('{"response": "qweqweqwe"}')
            resp.headers['Content-Type'] = "application/json"
        return resp
    
@app.route('/upload', methods=['GET',"POST"])
def upload():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join("K:\\DEV\\PROJECTs\\ChestCancerDetectionCNN\\uploads", image.filename))
            result = alexNet.predict("uploads/"+image.filename)
            http = ""
            if(result=="Adenocarcinoma"):
                text = "Your lung cancer type: "
                http = "https://en.wikipedia.org/wiki/Non-small-cell_lung_carcinoma#Lung_adenocarcinoma"
            if(result=="Large cell carcinoma"):
                text = "Your lung cancer type: "
                http = "https://en.wikipedia.org/wiki/Non-small-cell_lung_carcinoma#Large-cell_lung_carcinoma"
            if(result=="Normal"):
                text = "Your lung is healthy"
                result = ""
                http = "#"
            if(result=="Squamous cell carcinoma"):
                text = "Your lung cancer type: "
                http = "https://en.wikipedia.org/wiki/Non-small-cell_lung_carcinoma#Squamous-cell_lung_carcinoma"
            
            return render_template('index.html', text=text, type=result, link=http)
        
    return render_template('index.html')


if __name__ == "__main__":
    TRAIN_DATA_PATH = "Data/train"
    VALID_DATA_PATH = "Data/valid"
    TEST_DATA_PATH  = "Data/test"
    BATCH_SIZE      = 16
    TARGET_SIZE     = (227,227)
    INPUT_SHAPE     = [227,227,3]
    
    class CNNModel:
        def __init__(self, input_shape=0, model_type='', model_path=None):
            self.model_type = model_type
            self.history = None
            self.history_accuracy = []
            self.history_val_accuracy = []
            self.history_loss = []
            self.history_val_loss = []
            self.input_shape = input_shape
            self.model = tf.keras.models.Sequential()
            if(model_path==None):
                self.model_path = 'savedModel/CNNModel.ckpt'
            else:
                self.model_path = model_path
            self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_path,save_weights_only=True,verbose=1)

        def fit(self,train_data, validation_data, epochs):
            self.history = self.model.fit(x = train_data, validation_data = validation_data, epochs = epochs, callbacks=[self.model_checkpoint])
            self.history_accuracy.extend(self.history.history['accuracy'])
            self.history_val_accuracy.extend(self.history.history['val_accuracy'])
            self.history_loss.extend(self.history.history['loss'])
            self.history_val_loss.extend(self.history.history['val_loss'])

        def plotAccuracy(self):
            if(self.history != None):
                print(self.history.history.keys())
                plt.plot(self.history_accuracy)
                plt.plot(self.history_val_accuracy)
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train','test'], loc='upper left')
                plt.show()
            
        def plotLoss(self):
            if(self.history != None):
                plt.plot(self.history_loss)
                plt.plot(self.history_val_loss)
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()

        def compareAccuracy(models):
            modelList = []
            for model in models:
                if(model.history != None):
                    plt.plot(model.history_accuracy)
                    modelList.append(model.model_type)
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(modelList, loc='upper left')
            plt.show()

        def compareLoss(models):
            modelList = []
            for model in models:
                if(model.history != None):
                    plt.plot(model.history_loss)
                    modelList.append(model.model_type)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(modelList, loc='upper left')
            plt.show()

        def predict(self, path):
            classes_dir = ["Adenocarcinoma","Large cell carcinoma","Normal","Squamous cell carcinoma"]
            image = tf.keras.utils.load_img(path, target_size=TARGET_SIZE)
            normalizedImage = tf.keras.utils.img_to_array(image)/255
            imageArray = np.array([normalizedImage])
            predtionResult = np.argmax(self.model.predict(imageArray))
            return classes_dir[predtionResult]

        def load(self,model_path=None):
            if(model_path==None):
                self.model.load_weights(self.model_path)
            else:
                self.model.load_weights(model_path)
    # Original AlexNet
    class AlexNet(CNNModel):
        def __init__(self,input_shape=0,model_path='savedModel/AlexNet.ckpt'):

            super().__init__(input_shape=input_shape,model_type="AlexNet",model_path=model_path)
            self.model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=11, activation='relu', strides=4, input_shape=input_shape))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
            self.model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
            self.model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'))
            self.model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'))
            self.model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(0.5))
            self.model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(0.5))
            self.model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics = ['accuracy'])      
    
    alexNet = AlexNet(INPUT_SHAPE)
    alexNet.load()
    
    app.run(debug=True)