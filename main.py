import re
import base64
from flask import Flask, render_template,request

import joblib
import cv2
import numpy as np
import csv

from datetime import datetime 

algorithm=joblib.load('digits_model_svm.sav')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('drawdigits.html')

@app.route('/predictdigits/', methods=['GET','POST'])
def predict_digits():
    parseImage(request.get_data())
    img=cv2.imread('output.png')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(8,8))
    flatten=np.reshape(resized,(1,64))

    scaled=np.array(flatten/255.0*15,dtype=np.int)
    result=algorithm.predict(scaled)
    return str(result) 

def crop(im):

    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)         
        if(i==1):
            return thresh1[y:y+h,x:x+w]
        i=i+1

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

@app.route('/update', methods=['GET','POST'])
def update():
    result=request.form
    actual_label=int(result['actual'])

    now= datetime.now() #current date and time

    date_time= now.strftime("%m/%d/%Y, %H:%M:%S")

    img_name=date_time+'.png'

    img=cv2.imread('output.png')
    cv2.imwrite('data/'+img_name,img)
                            

    with open('data/dataset.csv','a') as file:
        writer=csv.writer(file)
        writer.writerow([img_name,actual_label])
    

    return render_template('drawdigits.html')


if __name__ == '__main__':
    app.run(debug=True)


    
    
