# Gaze Estimation

Estimating human gaze from natural eye images.

## Dataset



## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data preprocess
Extract images, scan them, to get bounding boxes and landmarks:
```bash
$ python extract.py
$ python pre_process.py
```

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage


### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

### Demo
```bash
$ python demo.py
```


Image | Aligned | Out | True |
|---|---|---|---|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_angle.jpg)|-0.4680,-20.1403,-52.9704|-2.1199,-21.3255,-53.1361|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_angle.jpg)|29.9972,-2.6437,-48.7752|25.0784,-0.0974,-51.5146|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_angle.jpg)|46.1205,-4.8502,-32.8676|45.1548,-5.7697,-34.7957|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_angle.jpg)|0.0907,34.9278,-45.5467|-2.4064,33.7071,-46.2663|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_angle.jpg)|-14.5307,-8.1151,-54.1164|-16.0543,-9.2991,-54.2075|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_angle.jpg)|-25.1123,5.9670,-51.1620|-28.5390,5.0019,-49.4291|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_angle.jpg)|4.2336,-15.4438,-54.2143|6.6807,-16.2491,-54.5341|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_angle.jpg)|36.7286,-8.5860,-42.0209|39.7002,-6.4171,-40.8061|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_angle.jpg)|35.9284,-23.6931,-38.9796|35.8156,-23.6345,-37.9699|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_angle.jpg)|-29.6480,12.5983,-47.0566|-28.5218,10.0440,-48.6670|