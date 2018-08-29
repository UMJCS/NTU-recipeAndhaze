import cv2
import os
import random
def GetFileList(dir, fileList):

    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList
namelist = GetFileList('./all',[]) # len 110000
list_len = len(namelist)
index_queue = [i for i in range(0,list_len)]
#print(index_queue[0])
random.shuffle(index_queue)
#print(index_queue[0])
count = 0
#levels = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
def save2file(file_name,context):
    fh = open(file_name,'a')
    fh.write(context)
    fh.close()

for index in index_queue:

    im_pth = namelist[index]
    underlineindex = im_pth.rfind('_')
    level = im_pth[underlineindex+1:-4]
    level = int(float(level)/0.05)
    name = im_pth[6:]
    

    if count<90000: # train
        save2file('train.txt','/train/'+ name +'\n')
        save2file('train_labels.txt',str(level)+'\n')
        save2file('train_paras.txt',im_pth[underlineindex+1:-4]+'\n')
        img = cv2.imread(im_pth)
        im_savepth = './train/'+ name
        cv2.imwrite(im_savepth,img)
    elif count>=9000 and count<10000: # valid
        save2file('valid.txt','/valid/'+ name +'\n')
        save2file('valid_labels.txt',str(level)+'\n')
        save2file('valid_paras.txt',im_pth[underlineindex+1:-4]+'\n')
        img = cv2.imread(im_pth)
        im_savepth = './valid/' + name
        cv2.imwrite(im_savepth, img)
    else: # test
        save2file('test.txt','/test/'+ name +'\n')
        save2file('test_labels.txt',str(level)+'\n')
        save2file('test_paras.txt',im_pth[underlineindex+1:-4]+'\n')
        img = cv2.imread(im_pth)
        im_savepth = './test/' + name
        cv2.imwrite(im_savepth, img)
    print("process: {} / {} ".format(count,list_len))
    count+=1
    




