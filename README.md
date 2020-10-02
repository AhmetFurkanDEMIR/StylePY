# StylePY

* With this web application, you can transfer styles between your images and videos in a simple interface.

* You will be able to create magnificent pictures with Neural style transfer.

* If I explain the purpose of the application in a simple way, we can say that it is to transfer the style of a picture to another picture, or we can say to transfer the style of a picture to another video.

* web application was written in flask, neural style transfer was written in pytorch.

* This application requires high GPU usage, if you do not have a good GPU, read my article titled Deploying flask project over Colab which I wrote earlier. https://github.com/AhmetFurkanDEMIR/Unix-and-Cloud-Computing/blob/master/GogleColab_Flask/README.md

### Use of the application

* First, go to the folder named models and run the code. (Since the size of the model (500 mb) is very large, you will download the model remotely with this code.)
code : 
```linux
bash ./download_models.sh
```

* Now let's install the modules required for the operation of this web application with pip.
code :
```linux
pip3 install -r requirements.txt
```

* Now everything is ok, we can run the application :).
code :
```linux
python3 main.py
```

* If you see the picture below, everything is fine, proceed to the link in the window that opens and enjoy the application.
![Screenshot_2020-10-02_11-32-26](https://user-images.githubusercontent.com/54184905/94903869-18f7b780-04a3-11eb-874b-9562fbd8afc5.png)

### Application images

![Screenshot_20201002-114542_Chrome](https://user-images.githubusercontent.com/54184905/94905164-16965d00-04a5-11eb-9e6b-fe9dde6fc3be.jpg)

![Screenshot_2020-10-02_11-41-41](https://user-images.githubusercontent.com/54184905/94904812-907a1680-04a4-11eb-8aa7-9cf0bce492e7.png)

![Screenshot_2020-10-02_11-41-59](https://user-images.githubusercontent.com/54184905/94904822-94a63400-04a4-11eb-9cfe-f76cfb2eec73.png)

![Screenshot_2020-10-02_11-43-03](https://user-images.githubusercontent.com/54184905/94904796-8c4df900-04a4-11eb-9aca-e0d7f875fb80.png)

![Screenshot_2020-10-02_11-43-25](https://user-images.githubusercontent.com/54184905/94904816-9243da00-04a4-11eb-80ad-0b8f52905ab3.png)

### Test results

* **Style Image**
 
![out1](https://user-images.githubusercontent.com/54184905/94905312-5a896200-04a5-11eb-8cdf-1e67bc064b4f.png)

![out1(1)](https://user-images.githubusercontent.com/54184905/94905334-61b07000-04a5-11eb-8ef8-3465f30dc7c9.png)

* **Style Video**

![asdddd-min](https://user-images.githubusercontent.com/54184905/94905674-edc29780-04a5-11eb-8078-11be4947aba9.gif)
