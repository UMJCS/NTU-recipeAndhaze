import json 
from collections import OrderedDict
import pickle,numpy
import random



def iter_dict(dic):
    for key in dic:
        print(key)
        # if isinstance(dic[key], dict):
        #     iter_dict(dic[key])

def Layer1Json(Pathname):
	with open(Pathname,'r', encoding='utf-8') as file:
		lines = file.readlines()[1:-2]
	#print(len(lines))
		for line in lines:
			load_data = json.loads(line[:-2])#,object_pairs_hook=OrderedDict)
			ingre_id_list.append(load_data['id'])
			temp_dict = {}
			# # temp_dict['ingredients'] = load_data['ingredients']
			# # temp_dict['url'] = load_data['url']
			temp_dict['partition'] = load_data['partition']
			# # temp_dict['title'] = load_data['title']
			# # temp_dict['instructions'] = load_data['instructions']
			ingre_info_list.append(temp_dict)
			#print(load_data.keys())

	file.close()
			#return load_data

def Layer2Json(Pathname):
	with open(Pathname,'r', encoding='utf-8') as file:
		lines = file.readlines()[1:-2]
	#print(len(lines))
		for line in lines:
			#print(line)
			#print("------------------")
			try:
				load_data = json.loads(line[:-2])#,object_pairs_hook=OrderedDict)
			except:
				load_data = json.loads[line]
			image_id_list.append(load_data['id'])
			#print(load_data['id'])
			temp_dict = {}
			temp_dict['images'] = load_data['images']
			if 'images' in load_data.keys():
				temp_dict['number'] = len(load_data['images'])
			image_info_list.append(temp_dict)
			#break
		#print(load_data['ingredients'])
		#print(load_data['ingredients'][0])
	file.close()
def DetJson(Pathname):
	with open(Pathname,'r', encoding='utf-8') as file:
		lines = file.readlines()
		print(json.loads(lines[0]).keys())
		#load_data = json.loads(lines)
		#print(load_data.keys())
	# for line in lines:
	# 	load_data = json.loads(line[:-2])#,object_pairs_hook=OrderedDict)
	# 	print(load_data.keys())
	# 	#print(load_data['ingredients'])
	# 	#print(load_data['ingredients'][0])
	# 	break


Layer1Path = '/home/minjie/NTU-recipeAndhaze/layer1.json'
Layer2Path = '/home/minjie/NTU-recipeAndhaze/layer2.json'
DetPath = '/home/minjie/NTU-recipeAndhaze/det_ingrs.json'

image_id_list = []
image_info_list = []
ingre_id_list = []
ingre_info_list = []

keys = open('val_keys.pkl','rb')
info = pickle.load(keys)
print(len(info))
RANDOM_Number = [random.randint(0,51128) for i in range(0,2)]
print(RANDOM_Number) 
try_list = [info[i] for i in RANDOM_Number]
print(try_list)
keys.close()
partition = 'val'
Layer1Json(Layer1Path)
Layer2Json(Layer2Path)
m = 0
q = 0
write_file = open('val.txt','a')
for i in try_list:
	print('----------------')
	#print(i)
	if i in image_id_list:
		index = image_id_list.index(i)
		#print(image_info_list[index])
		imageset = image_info_list[index]
		print(imageset)
		#print("image: "+str(index))
		for subdict in imageset['images']:
			imageName = subdict['id']
			imagePath = partition+'/' + imageName[0] + '/' + imageName[1] + '/' + imageName[2] + '/' + imageName[3] + '/' + imageName + ' '
			write_file.write(imagePath)
		m+=1
		write_file.write('\n')
	else:
		pass
		#print("No image id")
	if i in ingre_id_list:
		index = ingre_id_list.index(i)
		#print("ingre: "+ str(index))
		q+=1
		#print(ingre_info_list[index])
		#if ingre_info_list['partition'] == 'val'
	else:
		pass	
		#print("NO ingre id")
write_file.close()
print("Summary --------->")
print("image hit rate: "+str(m))
print("image hit rate: "+str(len(RANDOM_Number)))
print("ingre hit rate: "+str(q/len(RANDOM_Number)))

