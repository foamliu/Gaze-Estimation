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
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_angle.jpg)|0.0758,-0.3528,-0.9226|0.1150,-0.3388,-0.9338|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_angle.jpg)|-0.6256,-0.1121,-0.7706|-0.5700,-0.1475,-0.8083|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_angle.jpg)|0.2736,0.1576,-0.9499|0.2881,0.1396,-0.9474|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_angle.jpg)|0.0153,-0.3668,-0.9349|0.0276,-0.3739,-0.9270|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_angle.jpg)|0.0191,-0.0184,-1.0016|0.0379,-0.0488,-0.9981|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_angle.jpg)|-0.6786,0.0569,-0.7522|-0.6447,0.0332,-0.7637|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_angle.jpg)|-0.4399,0.2516,-0.8591|-0.4054,0.2669,-0.8743|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_angle.jpg)|0.2602,0.0399,-0.9845|0.2772,0.0692,-0.9583|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_angle.jpg)|-0.2465,0.1605,-0.9569|-0.2526,0.1354,-0.9580|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_angle.jpg)|0.2486,-0.2455,-0.9317|0.3022,-0.2460,-0.9210|