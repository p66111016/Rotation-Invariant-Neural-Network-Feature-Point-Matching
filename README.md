<h1 align="center">Rotation-Invariant-Neural-Network-Feature-Point-Matching</h1>

![invariant example](example/invariant_example.gif)
![original example](example/original_example.gif)

The examples of original SuperPoint and SuperGlue framework matching result (top). The results show that matching at angles exceeding 60 degrees significantly decreases the number of matched feature points. Our rotation-invariant method obtains many matched points even at large angles (bottom).




---
The inference codes for rotation invariant feature point matching. It takes as input a pair of images and return a matched points result.
Our approach is built upon the SuperPoint and SuperGlue framework, with several architectural modefication and employing alternative model traning strategies. Model code adapted from: https://github.com/vdvchen/SGMNet.

## 1. Download pre-trained model
* [Get pre-trained model in this link]
(https://drive.google.com/drive/folders/1d759SWfKwSFlZxCq6QjqoRN6vKuRUs3N?usp=sharing):Put pretrained model into folder "weight/"

## 2. Environment

- Prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Run the model
Run the code by following command. 

For matching feature points between a single pair of images.
  For example:
```python
python main.py 
--img1_path 'demo/demo_image/buiding_demo1.jpg' 
--img2_path 'demo/demo_image/buiding_demo2.jpg'
```
- If you want to match without using rotation invariant model (original superpoint&superglue)
```python
python main.py 
--config_path 'configs/sg_splight.yaml' 
--img1_path 'demo/demo_image/buiding_demo1.jpg' 
--img2_path 'demo/demo_image/buiding_demo2.jpg'
```

  For demonstrating feature point matching from multiple angles between a single pair of images.
For example:
```python
python rotation_sequence.py 
--img1_path 'demo/demo_image/buiding_demo1.jpg'  
--img2_path 'demo/demo_image/buiding_demo2.jpg'
```
- If you want to match without using rotation invariant model (original superpoint&superglue)
```python
python rotation_sequence.py 
--config_path 'configs/sg_splight.yaml' 
--img1_path 'demo/demo_image/buiding_demo1.jpg' 
--img2_path 'demo/demo_image/buiding_demo2.jpg'
```

## 4. References
* [SuperPoint](https://github.com/rpautrat/SuperPoint)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
