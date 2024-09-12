import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2

app = Flask(__name__)

# Set the path to the local folder to save uploaded images
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dic = {0 : 'diabetic_retinopathy', 1 : 'glaucoma', 2 : 'normal'}

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('upload.html')

img_size_x = 224
img_size_y = 224
model = load_model('model.h5')

def predict_label(img_path):

    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size_x,img_size_y)) 
    i = img_to_array(resized)/255.0
    i = i.reshape(1,img_size_x,img_size_y,1)
    predict_x=model.predict(i) 
    p=np.argmax(predict_x,axis=1)
    return dic[p[0]]

    # img = cv2.imread(img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # Assuming img_size_x and img_size_y are defined elsewhere in your script
    # resized = cv2.resize(gray, (img_size_x, img_size_y))
    
    # # Normalize the pixel values to the range [0, 1]
    # i = img_to_array(resized) / 255.0
    
    # # Convert single-channel image to three channels
    # i = np.stack((i,)*3, axis=-1)
    
    # # Ensure 3 channels for RGB images
    # i = i.reshape(1, img_size_x, img_size_y, 3)
    
    # # Make the prediction
    # predict_x = model.predict(i)
    
    # # Get the predicted label
    # p = np.argmax(predict_x, axis=1)
    
    # # Assuming dic is defined elsewhere in your script
    # return dic[p[0]]



@app.route("/upload", methods=["GET", "POST"])
def upload():
    p = None
    img_path = None
    if request.method == "POST" and 'photo' in request.files:
        # Get the uploaded file from the form data
        file = request.files['photo']

        # Save the file to the local folder
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_path = file_path

        p = predict_label(img_path)

    cp = str(p).lower() if p is not None else ""
    src = img_path if img_path is not None else ""
        

    return render_template('upload.html', cp=cp, src=src)




if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
