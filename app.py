from __future__ import print_function
import pickle
from flask import Flask,request
import numpy as np
from IPython import embed
from keras.models import load_model
model = load_model('char_m1_model.h5')
model.load_weights('char_m1_weights.h5')
maxlen=150

with open('charmap.pickle', 'rb') as handle:
    char_indices, indices_char = pickle.load(handle)

len_chars=len(char_indices)

def text2input(text_str):
    output = np.zeros((1, maxlen, len_chars), dtype=np.bool)
    for text_index, current_char in enumerate(text_str):
        if text_index < maxlen:
            output[0, text_index, char_indices[current_char]] = 1
    return output

# embed();
app = Flask(__name__)
print(model.predict(text2input('สวัสดี')))
@app.route("/")
def hello():
    return "Hello World!"

@app.route("/bait")
def bait_detect():
    text = request.args.get('text', '')
    model_result = model.predict(text2input(text))
    if model_result[0,0]>model_result[0,1]:
        return "Normal"
    else:
        return "Click bait!!!!!"



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)