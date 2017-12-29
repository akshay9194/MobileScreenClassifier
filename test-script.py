import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
from keras.preprocessing.image import img_to_array, load_img
import cv2

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
img = load_img('DemoModel.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
arrayresized = cv2.resize(x, (64,64))

inputarray = arrayresized[np.newaxis,...] # dimension added to fit input size

out = loaded_model.predict(inputarray)
print(out)
print(np.argmax(out,axis=1))
if np.argmax(out,axis=1) == 0:
    print("Highly-Broken")
if np.argmax(out,axis=1) == 1:
    print("Moderately-Broken")
if np.argmax(out,axis=1) == 2:
    print("Non-Broken")
y_classes = out.argmax(axis=-1)
print(y_classes)

