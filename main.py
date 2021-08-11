# Flask
# Bir şablonu oluşturmak için render_template() yöntemini kullanabilirsiniz
# Flask uygulamasını olusturup ayaga kaldırmak icin Flask(__name__) yöntemini kullanabilirsiniz
# Koddaki duruma gore on yuzde (HTML) hata veya uyari mesajları vermek icin flash methodunu kullanabilirsiniz
# istekleri kontrol etmek icin request methodu kullanılır
# jsonify; Bir application/json mimetype ile verilen bağımsız değişkenlerin JSON temsiliyle bir Yanıt oluşturur. Bu işlevin argümanları, dict yapıcısıyla aynıdır.
from flask import render_template, Flask, flash, request, jsonify

# path, isletim sistemi vb. icin
import os

# secilen bir dosyanin guvenli bir surumunu geri donderir
from werkzeug.utils import secure_filename

# farklı thredlerde farklı işler yaparak kodda dallanma sağlamak
from threading import Thread

# süre
import time

# resim sitili icin yazdigim fonskiyon
import image as im

# video sitili icin yazdigim fonskiyon
import video as vid

# video ve image stayle kisminda islem bittimi kontrolu icin refresh yapar
global asd
global asd0

asd = 0
asd0 = 0

# calisma alani dosyalari bu alana tasiyip bu alandan ceker
global path
path = os.getcwd()

# style isleminin Flask dan bagimsiz calismasini sağlamak
th = Thread()
finished = False
finished0 = False

# dosyalar
global content_file
global stil_file

app = Flask(__name__)

# upload turleri
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']

# upload yolu
app.config['UPLOAD_PATH'] = 'static/uploads'

# Şifreleme gerektiren herhangi bir şey (saldırganların kurcalamasına karşı güvenlik sağlamak için), gizli anahtarın ayarlanmasını gerektirir.
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# image style 'in koddan bagimsiz calismasini saglayan fonsiyon
def dataa():

    global finished
    im.asd()
    finished = True

# Video style 'in koddan bagimsiz calismasini saglayan fonsiyon
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

    # GET : sunucudan veri istemek için.
    # POST : işlenecek verileri sunucuya göndermek için.
    
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

            file = open("{}/booll/asd.txt".format(path),"w+")

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

            filea = open("{}/booll/asdd.txt".format(path), "w+")
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

            file = open("{}/booll/asd.txt".format(path),"w+")

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

            filea = open("{}/booll/asdd.txt".format(path), "w+")
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

    app.run(debug=True, host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))
