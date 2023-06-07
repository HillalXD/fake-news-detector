from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json

app = Flask(__name__)

model = load_model('Rnnmodel.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    
    sequences = tokenizer.texts_to_sequences(text)
    max_sequence_length = max([len(seq) for seq in sequences])
    
    vocab_size = len(tokenizer.word_index) + 1
    X = pad_sequences(sequences, maxlen=max_sequence_length)
    
    prediction = model.predict(X)

    if prediction[0] > .5:
        result = 'Real News'
    else:
        result = 'Fake News'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
