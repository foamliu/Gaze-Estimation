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


Image | True | Out | Plot |
|---|---|---|---|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_raw.jpg)|$(result_out_0)|$(result_true_0)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_raw.jpg)|$(result_out_1)|$(result_true_1)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_raw.jpg)|$(result_out_2)|$(result_true_2)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_raw.jpg)|$(result_out_3)|$(result_true_3)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_raw.jpg)|$(result_out_4)|$(result_true_4)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_raw.jpg)|$(result_out_5)|$(result_true_5)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_raw.jpg)|$(result_out_6)|$(result_true_6)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_raw.jpg)|$(result_out_7)|$(result_true_7)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_raw.jpg)|$(result_out_8)|$(result_true_8)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_raw.jpg)|$(result_out_9)|$(result_true_9)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_angle.jpg)|