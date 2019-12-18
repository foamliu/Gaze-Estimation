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
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_raw.jpg)|look_vec: 0.2209,0.4703,-0.8544<br>pupil_size: 0.0233|look_vec: 0.1917,0.4689,-0.8622<br>pupil_size: 0.0301|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_raw.jpg)|look_vec: 0.1296,-0.0196,-0.9914<br>pupil_size: -0.3122|look_vec: 0.2225,0.0327,-0.9744<br>pupil_size: -0.2559|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_raw.jpg)|look_vec: -0.6035,0.0833,-0.7930<br>pupil_size: 0.2096|look_vec: -0.6263,0.0704,-0.7764<br>pupil_size: 0.2214|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_raw.jpg)|look_vec: -0.5143,-0.2162,-0.8299<br>pupil_size: 0.0997|look_vec: -0.5563,-0.2640,-0.7879<br>pupil_size: 0.0541|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_raw.jpg)|look_vec: 0.3604,0.0302,-0.9323<br>pupil_size: 0.1686|look_vec: 0.3937,0.0203,-0.9190<br>pupil_size: 0.2245|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_raw.jpg)|look_vec: 0.6021,0.6270,-0.4943<br>pupil_size: 0.1146|look_vec: 0.5902,0.6401,-0.4918<br>pupil_size: 0.0198|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_raw.jpg)|look_vec: -0.0691,-0.2175,-0.9736<br>pupil_size: -0.0675|look_vec: -0.0955,-0.1726,-0.9804<br>pupil_size: -0.0526|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_raw.jpg)|look_vec: -0.4191,0.3765,-0.8262<br>pupil_size: -0.3669|look_vec: -0.3618,0.3485,-0.8646<br>pupil_size: -0.2192|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_raw.jpg)|look_vec: 0.0711,-0.1398,-0.9876<br>pupil_size: 0.2395|look_vec: 0.0902,-0.1491,-0.9847<br>pupil_size: 0.2621|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_angle.jpg)|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_raw.jpg)|look_vec: 0.3328,-0.4387,-0.8347<br>pupil_size: -0.2609|look_vec: 0.3543,-0.4874,-0.7981<br>pupil_size: -0.2140|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_angle.jpg)|