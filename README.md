# NTU-recipeAndhaze
## Image to recipe and coral && Haze Detection (later)
Summer research summary repo

#### Server : 172.21.64.58
#### Data path/media/external/foodAIDisk/data.csail.mit.edu/im2recipe/

### 2018.8.2 Take project and dataset 
#### 1. Reference website http://im2recipe.csail.mit.edu/
#### 2. Figure out data strcture: 2 layer json ... first for information and second for finding image ---> connect by id
#### 3. Clean up dataset
#### 4. Read paper http://im2recipe.csail.mit.edu/im2recipe.pdf
#### 5. Server environment cuda cudnn pytorch

### Weekend 

### 2018.8.6
#### 1. Figure out json inforamtions &radic;  See json_structure.txt
##### Det_ingrs.json  -> only one string should divide then convert to json information 
##### Layer1.json -> dict_keys(['ingredients', 'url', 'partition', 'title', 'id', 'instructions'])  &radic;
url stands for website contain ingredients,title,instructions
##### Layer2.json -> dict_keys(['id', 'images']) image = image_id + url(more than one sometime) &radic;

#### 2. Try to clean dataset and connect json information accordingly to image  &radic;
##### How to find the image from json information &radic;
#### 3. Finish reading paper &radic;
#### 4. Find relative works 

### 2018.8.7
#### 1. Figure out github code and dataset structure &radic;
#### 2. ~~Fail to download dataset (92 Gbytes) to remote severe~~
#### 3. Open pkl and mbd dataset &radic;

TodoList 
1. ~~Read layer2.json to rule out non-image id and summary image number for each id~~  &radic;
2. ~~Verify id according pair~~ &radic;

### 2018.8.8
#### 1. Succeed in linking train/val/test keys with dataset imagepath &radic;
#### 2. Write standary image path .txt for training in DeepLearning network input &radic;
#### 3. Read relative paper for thier usage of dataset  &radic;
#### 4. Succeed in reading all infomation in layer1/layer2 JSON &radic;

### 2018.8.9 festival off

### 2018.8.10 
#### 1. Generate val.txt/test.txt for Densenet trainning &radic;
#### 2. Relative paper
#### 3. New dataset food-101/ingredient101/recipe5K
#### 4. Clean dataset and find construction for training

### 2018.8.12 
#### 1.Generate all image path and labels for training in Resnet &radic;
#### 2.Modify resnet structure and dataload &radic;

### 2018.8.14 
#### 1. link food-101 dataset with ingredient-101 dataset (ingredient and image information combine) &radic;
#### 2. Code PathGenerator.py ImageGenerator.py VectorGenerator.py  &radic;
#### 3. Training Resnet18 (On going... 

### 2018.8.15
#### 1.Finish Checking dataset &radic;
#### 2. Finish modify resnet code &radic;
#### Training !!!! Loss initial: 10.8 &radic;

### 2018.8.16
#### 1. Resnet Loss to 6 nohup exp.log &radic;
#### 2. Read haze paper and IEEE template &radic;
#### 3. Other dataset &radic;

### 2018.8.17-2018.8.20
#### 1. Trainning Resnet epoches &radic; failed --> due to wrong index arrangement
#### 2. Download RESIDE-b dataset (2061 origin pics and over 70000 haze pics with depth information) &radic;
#### 3. Generate and implement paper method to different synthesize haze image by hand &radic;

### 2018.8.20 
#### 1. Debug and retrain resnet epoches ongoing (seem reasonable so far Loss: 1.5 --> see correct.log) 
#### 2. Augment haze dataset wiht different haze level -> (goal over 100000 pics) 
#### 3. Find way to generate haze images without depth image from real world single out-door images


### 2018.8.21
#### 1. Finish training resnet on binary task -> see logs/0.0001.log && 0.0005.log lr = 0.0001 && 0.0005
#### 2. Training resnet and Vgg on reconsturction task --> ongoing --> lr = 0.0001 save weights per epoch

### 2018.8.22
#### Experience 1: resnet lr = 1e-4 nonvalid epoch 7 ->
  ##### Test set: Average Top1_Accuracy_V:0.7784950495049505 | Average Top5_Accuracy_V:0.9411881188118811 -> weights && logs see /exps/exp0
#### Experience 2: vgg lr = 1e-4 nonvalid epoch 8 ->
  ##### Test set: Average Top1_Accuracy_V:0.7787722772277228 | Average Top5_Accuracy_V:0.9397227722772277 -> weights && logs see /exps/exp1
#### Experience 3: resnet lr = 5e-4 nonvalid epoch 7 -> on running epoch1 = 0.5359 epoch2 = 0.5986 *epoch 3 = 0.7413
#### Experience 4: resnet lr = 3e-4 nonvalid epoch 7 -> on running epoch1 = 0.6179 epoch2 = 0.6782 epoch3 = 0.76396  up but 3-6 no change(up and down)
#### Experience 5: resnet lr = 1e-4 valid epoch 7 -> on running epoch1 = 0.675 epoch2 = 0.7245 epoch3 =
#### Experience 6: resnet lr = 2e-4 valid epoch 12 -> on running epoch1 = 0.6328 epoch2 = 0.6921 epoch3 = 0.7735 epoch4 = 0.7761

#### Best now experience 9 : resnet lr = 1e-4 valid epoch 10 with predict_V * 8 -> epoch1 = 0.7075 epoch2 = 0.7363 epoch3 = 0.7883 epoch4 = 0.790059 epoch5 = 0.7885 epoch6 = 0.78942 -> epoch 10 change lr too high?

### 2018.8.22 - 2018.8.26
#### Haze dataset http://vcc.szu.edu.cn/research/2017/RSCM & https://drive.google.com/file/d/0B9H84JPfo-KtV1R5SF94ZV94Ymc/view
#### Generate 10 levels of synthesize haze photos also labels.txt imgpath.txt
#### Generate origin and 256 resize dataset --> publish??

### 2018.8.27
#### Add ingredient info to resnet and run with exp13 10 epoches and lr = 1e-4
#### Add ingredient info to vgg

### 2018.8.28
#### exp13-add ingredient  -> epoch 8 = 0.818178 epoch9* = 0.81932
#### exp16-retrain-with-exp13-epoch8 --->  epoch 5* = Top1_Accuracy_V:0.8203564356435643

### 2018.8.29
#### Haze cloudy dataset exp1 -> 1e-4 --> epoch7* lr = 1-e6  Average Top1_Accuracy_V:0.9872 | Average Top5_Accuracy_V:1.0 
#### retrain use epoch10 --> lr = 1e-8 0.9866
#### exp2 validation perportion 7:3 -->  lr = 1e-5 --> epoch5* Average Top1_Accuracy_V:0.8815 
#### Haze sunny dataset exp1 -> 1e-5 --> epoch8* lr = 1e-07 Average Top1_Accuracy_V:0.9400602085797226
#### exp2 7:3 validation --> epoch10* lr = 1e-08 Average Top1_Accuracy_V:0.8972691108482959

### 2018.8.30-8.31
#### Test and modify loss function 
#### --> 1. R value as paper--> epoch8 = Average Top1_Accuracy_V:0.9837
#### --> 2. Normal distribution curve function--> Average Top1_Accuracy_V:0.8802
