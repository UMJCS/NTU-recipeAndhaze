import os
import cv2
import numpy as np
import h5py
import math
from PIL import Image
desired_size = 256
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
def darkGenerator(BGRimg,img_name):
    B = BGRimg[:, :, 0]
    G = BGRimg[:, :, 1]
    R = BGRimg[:, :, 2]
    h = len(B)
    w = len(B[0])
    dark_channel = np.zeros((h, w, 1), np.uint8)

    h_valid_range = [i for i in range(3, h - 3)]
    w_valid_range = [i for i in range(3, w - 3)]
    V = 0
    V_pos = [0, 0]
    for i in range(0, h):
        for j in range(0, w):
            low_i = max(i - 3, 0)
            up_i = min(i + 3, h - 1)
            low_j = max(j - 3, 0)
            up_j = min(j + 3, w - 1)
            B_r = []
            G_r = []
            R_r = []
            for wi in range(low_i, up_i + 1):
                for wj in range(low_j, up_j + 1):
                    B_r.append(B[wi][wj])
                    G_r.append(G[wi][wj])
                    R_r.append(R[wi][wj])
            # break
            b_min = np.min(B_r)
            g_min = np.min(G_r)
            r_min = np.min(R_r)
            all_min = min(b_min, g_min, r_min)
            dark_channel[i][j] = all_min
            if all_min > V:
                V = all_min
                V_pos[0] = i
                V_pos[1] = j
    #cv2.imwrite(RootPath+'/dark/'+img_name+"_dark.jpg", dark_channel)
    A_B = B[V_pos[0]][V_pos[1]]
    A_G = G[V_pos[0]][V_pos[1]]
    A_R = R[V_pos[0]][V_pos[1]]
    A = np.zeros((h,w,3),dtype='uint8')
    A[:,:,0] = A_B
    A[:, :, 1] = A_G
    A[:, :, 2] = A_R
    return A


def cut(im_pth):
    print(im_pth)
    im = Image.open(im_pth)
    (x,y) = im.size
    #cut_side = max(old_size)
    #index = old_size.index(cut_side)
    if x > y:
        region = (int(x / 2 - y / 2), 0, int(x / 2 + y / 2), y)
        crop_img = im.crop(region)
        #crop_img.save(self.outfile)
    elif x < y:
        region = (0, int(y / 2 - x / 2), x, int(y / 2 + x / 2))
        crop_img = im.crop(region)
        #crop_img.save(self.outfile)
    else:
        crop_img = im

    crop_img.thumbnail((desired_size, desired_size))
    return crop_img
    #crop_img.save(im_pth)
namelist = GetFileList('./cloudy',[])
print(len(namelist))
RootPath = os.getcwd()
#beta = 0.5
betas = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
count = 0
rounds = 0
for im_pth in namelist:

    index =im_pth.rfind('\\')
    img_name = im_pth[index+1:-4]
    depinfoPath = '0987.mat'
    #depthPath = RootPath + '/vis_depth_map/' + img_name + '.jpg'
    originPath = im_pth
    origin_img = cv2.imread(originPath)

    #origin_img = cut(im_pth)
    #origin_img = np.array(origin_img)
    # new_origin_img = np.zeros((len(origin_img),len(origin_img[0]),3),dtype='uint8')
    # new_origin_img[:,:,0] = origin_img[:,:,2]
    # new_origin_img[:,:,1] = origin_img[:,:,1]
    # new_origin_img[:,:,2] = origin_img[:,:,0]
    # origin_img = new_origin_img
    img_shape = (len(origin_img),len(origin_img[0]))
    #depth_img = cv2.imread(depthPath)
    depth_info = h5py.File(depinfoPath)
    depth_info = np.array(depth_info['depth'][:]).transpose()
    # depth_shape = (len(depth_info),len(depth_info[0]))
    # depth_info = depth_info[0:256,197:453]
    # depth_shape = (len(depth_info),len(depth_info[0]))
    depth_info = cv2.resize(depth_info,(img_shape[1],img_shape[0]))
    A = darkGenerator(origin_img, img_name)
    for beta in betas:
        print("=======> beta: " + str(beta))
        up = (-beta * np.array(depth_info))
        tx = np.zeros((len(origin_img),len(origin_img[0]),3),dtype='float64')
        txx = np.zeros((len(origin_img),len(origin_img[0]),3),dtype='float64')
        for i in range(0,len(up)):
            for j in range(0,len(up[0])):
                tx[i][j][:] = math.exp(up[i][j])
                txx[i][j][:] = 1-math.exp(up[i][j])
        part1 = origin_img * tx
        part2 = A * txx
        Ix = part1+part2
        outputPath = RootPath+'/result/'+str(beta)+ '/' + img_name+'_'+ str(beta) + '.jpg'
        allPath = RootPath+'/result/all/' + img_name + '_' + str(beta) + '.jpg'
        cv2.imwrite(outputPath,Ix)
        cv2.imwrite(allPath,Ix)
        #print("One beta finished {}/{}".format(round, len(betas)))
        #rounds += 1
    if count%1 == 0:
        print("ratio: {}/{} ".format(count,len(namelist)))
        count += 1
