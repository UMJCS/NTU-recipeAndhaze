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

#### 2. Try to clean dataset and connect json information accordingly to image
##### How to find the image from json information 
#### 3. Finish reading paper &radic;
#### 4. Find relative works 

### 2018.8.7
#### 1. Figure out github code and dataset structure
#### 2. Fail to download dataset (92 Gbytes) to remote severe
#### 3. Open pkl and mbd dataset 
