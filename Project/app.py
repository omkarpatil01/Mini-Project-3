
from fileinput import filename
from unittest import result
from django.shortcuts import redirect
from flask import Flask, request, render_template
import keras
import librosa
import numpy as np

loaded_model = keras.models.load_model('Emotion1.h5')

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():

    if request.method =="POST":
       

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            def convert_class_to_emotion(pred):
        
                label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

                for key, value in label_conversion.items():
                    if int(key) == pred:
                       label = value
                return label    
            data, sampling_rate = librosa.load(file)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=1)
            x = np.expand_dims(x, axis=0)
            predictions = np.argmax(loaded_model.predict(x), axis=-1)
            result =  convert_class_to_emotion(predictions)
            file = '../static/'+result+ '.jpg'
            print(file)       
            return render_template('index.html', transcript=result,filename=file)

    return  render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)

