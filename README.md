# GAN
Demo&amp;Learning
Environment: Anaconda 3.6.4 + RTX2070 + Tensorflow 1.14.0 (CPU & GPU) + Cuda10+cuDNN+Keras
DATA
===========================================================================================
Extra Data Set Download :
https://drive.google.com/file/d/1PIMW6Cv2r4X95nwCqZ14AcgJszm5nMk3/view

Reference:
============================================================================================
https://github.com/CodePlay2016/GAN_learn/tree/master/HW3
https://github.com/d31003/MLDS_2019spring/tree/master/HW3/3-1

Different GAN with keras: https://github.com/eriklindernoren/Keras-GAN

Result(GAN):
============================================================================================
it so difficulty to training (vanishing gradient)& it is easily  mode collapse
![image](https://user-images.githubusercontent.com/20764935/62004405-4a664c00-b157-11e9-836a-b1becd6026c2.png)
![image](https://user-images.githubusercontent.com/20764935/62004412-61a53980-b157-11e9-92f7-0f941fda4be2.png)
![image](https://user-images.githubusercontent.com/20764935/62004413-64a02a00-b157-11e9-8196-384791d8d9de.png)
![image](https://user-images.githubusercontent.com/20764935/62004415-67028400-b157-11e9-89bc-94eec9d0b479.png)

Result(WGAN):
=============================================================================================
solve the problem "vanishing gradient" & "mode collapse"

different to GAN:

1.remove the sigmoid in the last layer in D

2.remove log

3.setting the weight between -a and a

4.using RMSProp,SGD and so on , dont use momentum or Adam

![image](https://user-images.githubusercontent.com/20764935/62004997-5a822980-b15f-11e9-863b-87016976eeba.png)
![image](https://user-images.githubusercontent.com/20764935/62004999-5d7d1a00-b15f-11e9-9db0-adf3270e0f0a.png)
![image](https://user-images.githubusercontent.com/20764935/62005000-60780a80-b15f-11e9-80ea-605ab23bacae.png)
![image](https://user-images.githubusercontent.com/20764935/62005009-78e82500-b15f-11e9-8a03-2b3d9215eccc.png)

Result(CGAN):
=============================================================================================
add the label with noise to training :

1orange hair blue eyes 

2red hair orange eyes

3blue hair blue eyes

4green hair pink eyes

5blonde hair green eyes

![image](https://user-images.githubusercontent.com/20764935/62018161-20129e00-b1ec-11e9-9441-9d7ff2bbdab5.png)
![image](https://user-images.githubusercontent.com/20764935/62018230-65cf6680-b1ec-11e9-9c9e-a545c649f927.png)

