import pickle
from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Helper as p
import os
import cv2

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET'])
def return_index():
    return render_template('index.html')

@app.route("/classify_text", methods=['GET', 'POST'])
def text_classifier():
    print("in fn")
    if request.method == "POST":
        sent = request.form['sentence']
        print(sent)
        classm = ''
        if sent:
            clean = p.preprocess(sent)
            predict = p.text_classification(clean)
            print(round(predict[0][0]*100 , 2))
            if predict > 0.75:
                classm = 'Very Offensive'
                cssclass = "progress red"
            elif predict > 0.55:
                classm = 'Mildy Offensive'
                cssclass = 'progress yellow'
            else:
                classm = 'Not Offensive'
                cssclass = 'progress green'
            
            return render_template('index.html', result = round(predict[0][0]*100 ,2), classmsg = classm, csscl = cssclass)
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

@app.route("/fileupload", methods=['POST'])
def upload_file():
   if request.method == 'POST':
        if 'f1' not in request.files:
            print('there is no file1 in form!')
        print(request.files)
        f1 = request.files['f1']
        f1.save(f1.filename)
        img = cv2.imread(f1.filename)
        predict = p.image_classification(img)
        print(predict)
        if predict > 0.75:
               classm = 'Very Offensive'
               cssclass = "progress red"
        elif predict > 0.55:
            classm = 'Mildy Offensive'
            cssclass = 'progress yellow'
        else:
            classm = 'Not Offensive'
            cssclass = 'progress green'
            
        return render_template('index.html', result = round(predict[0][0]*100 ,2), classmsg = classm, csscl = cssclass)

@app.route("/ocrupload", methods=['POST'])
def ensemble():
    if request.method == 'POST':
        if 'f2' not in request.files:
                print('there is no file1 in form!')
        print(request.files)
        f2 = request.files['f2']
        f2.save(f2.filename)
        img = cv2.imread(f2.filename)
        txt = p.OCR(img)
        cleaned = p.preprocess(txt)
        p1 = p.text_classification(cleaned)[0][0]
        p2 = p.image_classification(img)[0][0]
        print(p1)
        print(p2)
        predict = (p1 + p2)/2
        print(predict)
        if p1 > 0.75 and p2 > 0.75:
               classm = 'Very Offensive based on both text and image'
               cssclass = "progress red"
        elif p1 > 0.75 and p2 <= 0.75:
            classm = 'Offensive based on text'
            cssclass = 'progress red'
        elif p1 <= 0.75 and p2 > 0.75:
            classm = 'Offensive based on image'
            cssclass = 'progress red'
        elif p1 > 0.5 and p2 > 0.5:
            classm = 'Mildly Offensive based on both text and image'
            cssclass = 'progress yellow'
        elif p1 > 0.5 and p2 <= 0.5:
            classm = 'Mildly Offensive based on text'
            cssclass = 'progress yellow'
        elif p1 <= 0.5 and p2 > 0.5:
            classm = 'Mildly Offensive based on image'
            cssclass = 'progress yellow'
        else:
            classm = 'Not Offensive'
            cssclass = 'progress green'
            
        return render_template('index.html', result = round(max(p1,p2)*100, 2), classmsg = classm, csscl = cssclass)
        

if __name__ == "__main__":
    app.run(host="localhost", debug=True, use_reloader=False)