from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

# Load the trained AI model
model = tf.keras.models.load_model("plant_disease_model.h5")
class_labels = ["Apple___Apple_scab", "Corn_(maize)__Common_rust_", "apple(healthy)", "blueberry(healthy)" , "cherry(healthy)", "cherry(powdery mildew)", "corn(maize)cercospora_leafspot" ,"corn(maise) Common rust" ,"corn(maize) healthy",  "northern lead blight", "black rot" , "grape esca(black measles)" ,"grape(healthy)", "grape(leaf blight)" ,"orange(haunglongbing)", "peach baterial spot", "peach healthy", "pepper bell bacterial spot", "potato early blight", "potato healthy", "potato late blight", "rasberry healthy",  "soybean healthy" ,"squash powdery mildew" ,"strawberry healthy" ,"strawberry leaf scorch", "tomato bacterial spot" ,"tomato early blight" ,"tomato healthy" ,"tomato late blight" ,"tomato yellow leaf mold"," tomato septoria leaf spot","tomato spider mites two","tomato target spot","tomato mosaic virus","tomato yellow lead curl virus"] 

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = image.load_img(file, target_size=(150, 150))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    if predicted_class < len(class_labels):
        result = {"class": class_labels[predicted_class], "confidence": float(predictions[0][predicted_class])}
    else:
        result = {"error": "Prediction index out of range!"}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
    
