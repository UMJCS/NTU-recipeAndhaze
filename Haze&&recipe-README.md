Recipe Project:
Location: 
Without ingredient --> /media/external/foodAIDisk/smalldateset --> see resnet-disentangle-reconstruction-valid.py
	train/vaild/test.txt contain relative image path 
	train/valid/test_label.txt contain image class label
With ingredient --> /media/external/foodAIDisk/multimodal --> see resnet-disentangle-multimodal.py
	train/vaild/test.txt contain relative image path 
	train/valid/test_label.txt contain image class label
	foodtype_vectors.txt contain each label a sparse vector with all ingredient words in vector_order.txt --> see origin ingredients in ./ingredients-101/annotations/ingredients_simplified.txt  the same order as label and image path

Haze Project:
Location:  Root path --> /media/menglei/
2 dataset (cloudy and sunny)  2 different validation propertion (9:1:1 ana 7:3:1) 
./{sunny,cloudy}73/ -> means train data : valid data = 7:3
./{SUNNY,Root path} --> means train data : valid date = 9:1

see code resnet-haze-binary.py
Every dataset foulder all have {train/test/valid}.txt --> related image path
{train/test/valid}_labels.txt --> haze level labels 0->10  last one might cause read problem, now add by hand with the last number (Base 10 error)
{train/test/valid}_paras.txt --> origin haze level parameter 0->5 each 0.5 interval --> but not use here

./logs saved some experiments logs
./exps save some experiments weights can retrain
