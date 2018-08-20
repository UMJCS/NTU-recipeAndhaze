import os
import cv2
import numpy as np
import h5py
import math
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
    cv2.imwrite(RootPath+'/dark/'+img_name+"_dark.jpg", dark_channel)
    A_B = B[V_pos[0]][V_pos[1]]
    A_G = G[V_pos[0]][V_pos[1]]
    A_R = R[V_pos[0]][V_pos[1]]
    A = np.zeros((h,w,3),dtype='uint8')
    A[:,:,0] = A_B
    A[:, :, 1] = A_G
    A[:, :, 2] = A_R
    return A
namelist = GetFileList('./../clear',[])
print(len(namelist))
RootPath = os.path.abspath(os.path.join(os.getcwd(), ".."))
#beta = 0.5
betas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
count = 0
rounds = 0
for im_pth in namelist:

    index =im_pth.rfind('\\')
    img_name = im_pth[index+1:-4]
    depinfoPath = RootPath + '/depth/' + img_name + '.mat'
    depthPath = RootPath + '/vis_depth_map/' + img_name + '.jpg'
    originPath = im_pth
    origin_img = cv2.imread(originPath)
    depth_img = cv2.imread(depthPath)
    depth_info = h5py.File(depinfoPath)
    depth_info = np.array(depth_info['depth'][:]).transpose()
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




