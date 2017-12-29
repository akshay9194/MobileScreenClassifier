#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request,flash,redirect,url_for
from werkzeug import secure_filename
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
from keras.preprocessing.image import img_to_array, load_img
import cv2
#for regular expressions, saves time dealing with string data
import re
import urllib.request as urllib2
import uuid
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#settings for upload folder
UPLOAD_FOLDER = os.path.abspath("./static/Uploads")
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))
	
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
            
@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
      response = 0
      if request.method == 'POST':
        if request.form['myBtn'] == 'Upload & Predict':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                response = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                imgNm = file.filename  
        
        if request.form['myBtn'] == 'Get & Predict':
            imgNm = str(uuid.uuid4())+'.jpg'
            with open(os.path.join(app.config['UPLOAD_FOLDER'], imgNm), 'wb') as f:
                f.write(urllib2.urlopen(request.form['URL']).read())
            response = predict(os.path.join(app.config['UPLOAD_FOLDER'], imgNm))
            
        templateData = {
            'title' : 'HELLO!',
            'response': response,
            'imgLoc' : imgNm
            }
        return render_template('index.html', **templateData)
          
#@app.route('/predict/',methods=['GET','POST'])
def predict(imgData):
    #imgData = request.get_data()
    #convertImage(imgData)
    print(imgData)
    print("debug")
    img = load_img(imgData)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    arrayresized = cv2.resize(x, (64,64))
    inputarray = arrayresized[np.newaxis,...] # dimension added to fit input size
    
    print("debug2")
    #in our computation graph
    with graph.as_default():
        #perform the prediction
        out = model.predict(inputarray)
        print(out)
        print(np.argmax(out,axis=1))
        if np.argmax(out,axis=1) == 0:
            response = "Highly-Broken"
    
        if np.argmax(out,axis=1) == 1:
            response = "Moderately-Broken"

        if np.argmax(out,axis=1) == 2:
            response = "Non-Broken"

        print("debug3")
        #convert the response to a string
        #response = np.array_str(np.argmax(out,axis=1))
        return response

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 6777))
	#run the app locally on the givn port
	app.run(host='127.0.0.1', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
