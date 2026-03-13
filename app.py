import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('mobilenet.keras')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save file to static folder to show it on UI
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Predict
            img = tf.keras.utils.load_img(filepath, target_size=(160, 160))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            prediction = model.predict(img_array)[0][0]
            label = "Dog" if prediction > 0.5 else "Cat"

            return render_template('index.html', label=label, img_path=filepath)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
