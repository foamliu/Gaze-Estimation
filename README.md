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
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/0_img.jpg)|-0.3409,-0.1258,-0.9238|-0.2904,-0.1293,-0.9481|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/1_img.jpg)|-0.5860,0.0668,-0.7822|-0.6098,0.0796,-0.7886|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/2_img.jpg)|-0.0899,0.4285,-0.9038|-0.1119,0.4240,-0.8988|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/3_img.jpg)|0.0890,0.2886,-0.9421|0.1115,0.3238,-0.9395|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/4_img.jpg)|-0.2410,0.0171,-0.9714|-0.2878,0.0072,-0.9577|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/5_img.jpg)|0.1610,0.5602,-0.7974|0.2274,0.5526,-0.8018|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/6_img.jpg)|-0.4280,-0.0820,-0.8856|-0.4890,-0.0867,-0.8680|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/7_img.jpg)|0.4681,0.5640,-0.6807|0.4459,0.5740,-0.6868|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/8_img.jpg)|0.0628,-0.0323,-0.9857|0.1441,0.0058,-0.9896|
|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Gaze-Estimation/raw/master/images/9_img.jpg)|0.5952,-0.4515,-0.6437|0.6167,-0.4434,-0.6505|