from flask import render_template, Flask, flash, redirect, session, request, render_template, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
from threading import Thread
import sys
import time
import image as im
import video as vid
import time
global asd
global asd0

asd = 0
asd0 = 0

th = Thread()
finished = False
finished0 = False

global content_file
global stil_file

app = Flask(__name__)

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def dataa():

    global finished
    im.asd()
    finished = True

def data():

    global finished0
    vid.asdd()
    finished0 = True


@app.route("/")
def main():

    return render_template("index.html")


@app.route("/VideoStyle",methods = ['GET', 'POST'])
def VideoStyle():

    global content_file
    global stil_file

    sz = {"a":False, "b":False, "c":False, "d":False}

    
    if request.method == "POST":

        try:
        
            content_file = request.files['file1']
            stil_file = request.files['file']
            filename = secure_filename(content_file.filename)
            temp = filename.split(".")
            content_file.save(os.path.join(app.config['UPLOAD_PATH'], "content0."+temp[1]))

            filename = secure_filename(stil_file.filename)
            temp = filename.split(".")
            stil_file.save(os.path.join(app.config['UPLOAD_PATH'], "stil."+temp[1]))

        except:
            sz["a"] = True
            pass

        try:
            
            true = request.form['gender']

            file = open("../a/booll/asd.txt","w+")

            file.write(true)
            file.close()

        except:

            sz["b"] = True
            pass
            
        try:

            height = request.form['height']
            width = request.form['width']

            if (int(height)<100 or int(height) > 1000) or (int(width)<100 or int(width) > 1000):
                
                sz["c"] = True
                pass

            filea = open("../a/booll/asdd.txt", "w+")
            filea.write(height +","+ width)
            filea.close()

        except:
            
            sz["d"] = True
            pass

        flaa = False

        if sz["a"] == True:

            flash("Image data is missing or extensions of data are incorrect")
            flaa = True

        if sz["b"] == True:

            flash("Please select whether to have color protection")
            flaa = True

        if sz["c"] == True:

            flash("width and height cannot be less than 100 and greater than 1000")
            flaa = True

        if sz["d"] == True:

            flash("Please enter height and width in data type int")
            flaa = True

        if flaa == True:

            return render_template("/VideoStyle.html")
        
        global th0
        global finished0

        finished0 = False
        th0 = Thread(target=data)
        th0.start()

        return render_template("/VideoStyle0.html")

    
    else:

        return render_template("/VideoStyle.html")



@app.route("/ImageStyle",methods = ['GET', 'POST'])
def ImageStyle():

    global content_file
    global stil_file

    sz = {"a":False, "b":False, "c":False, "d":False}

    
    if request.method == "POST":

        try:
        
            content_file = request.files['file1']
            stil_file = request.files['file']
            filename = secure_filename(content_file.filename)
            temp = filename.split(".")
            content_file.save(os.path.join(app.config['UPLOAD_PATH'], "content."+temp[1]))

            filename = secure_filename(stil_file.filename)
            temp = filename.split(".")
            stil_file.save(os.path.join(app.config['UPLOAD_PATH'], "stil."+temp[1]))

        except:
            sz["a"] = True
            pass

        try:
            
            true = request.form['gender']

            file = open("../a/booll/asd.tx","w+")

            file.write(true)
            file.close()

        except:

            sz["b"] = True
            pass
            
        try:

            height = request.form['height']
            width = request.form['width']

            if (int(height)<100 or int(height) > 6000) or (int(width)<100 or int(width) > 6000):
                
                sz["c"] = True
                pass

            filea = open("../a/booll/asdd.txt", "w+")
            filea.write(height +","+ width)
            filea.close()

        except:
            
            sz["d"] = True
            pass

        flaa = False

        if sz["a"] == True:

            flash("Image data is missing or extensions of data are incorrect")
            flaa = True

        if sz["b"] == True:

            flash("Please select whether to have color protection")
            flaa = True

        if sz["c"] == True:

            flash("width and height cannot be less than 100 and greater than 6000")
            flaa = True

        if sz["d"] == True:

            flash("Please enter height and width in data type int")
            flaa = True

        if flaa == True:

            return render_template("/ImageStyle.html")
        
        global th
        global finished

        finished = False
        th = Thread(target=dataa)
        th.start()

        return render_template("/ImageStyle0.html")

    
    else:
        
        return render_template("/ImageStyle.html")


@app.route('/result',methods = ['GET', 'POST'])
def result():
    global asd

    asd += 1
    """ Just give back the result of your heavy work """
    return render_template("/ImageStyle1.html",asd=asd)

@app.route('/result0',methods = ['GET', 'POST'])
def result0():
    global asd0

    asd0 += 1
    """ Just give back the result of your heavy work """
    return render_template("/VideoStyle1.html",asd0=asd0)

@app.route('/status')
def thread_status():
    """ Return the status of the worker thread """
    time.sleep(20)
    return jsonify(dict(status=('finished' if finished else 'running')))

@app.route('/status0')
def thread_status0():
    """ Return the status of the worker thread """
    time.sleep(20)
    return jsonify(dict(status=('finished' if finished0 else 'running')))

@app.route('/me')
def me():

    return render_template("/me.html")


if __name__ == "__main__":

    app.run(debug=True)
