import json 
from collections import OrderedDict

def iter_dict(dic):
    for key in dic:
        print(key)
        # if isinstance(dic[key], dict):
        #     iter_dict(dic[key])

def LayerJson(Pathname):
	with open(Pathname,'r', encoding='utf-8') as file:
	lines = file.readlines()[1:-1]
	#print(len(lines))
	for line in lines:
		load_data = json.loads(line[:-2])#,object_pairs_hook=OrderedDict)
		print(load_data.keys())
		#print(load_data['ingredients'])
		#print(load_data['ingredients'][0])
		break
def DetJson(Pathname)
with open(Pathname,'r', encoding='utf-8') as file:
	lines = file.readline()
	
	load_data = json.loads(lines)
	print(load_data.keys())
	# for line in lines:
	# 	load_data = json.loads(line[:-2])#,object_pairs_hook=OrderedDict)
	# 	print(load_data.keys())
	# 	#print(load_data['ingredients'])
	# 	#print(load_data['ingredients'][0])
	# 	break


Layer1Path = '/home/minjie/NTU-recipeAndhaze/layer1.json'
Layer2Path = '/home/minjie/NTU-recipeAndhaze/layer2.json'
DetPath = '/home/minjie/NTU-recipeAndhaze/det_ingrs.json'
DetJson(DetPath)
