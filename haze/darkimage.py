import cv2
import numpy as np
import copy
BGRimg = cv2.imread("test.jpeg")
# 178 * 178
B = BGRimg[:,:,0]
G = BGRimg[:,:,1]
R = BGRimg[:,:,2]
h = len(B)
w = len(B[0])
dark_channel = np.zeros((h,w,1), np.uint8)


h_valid_range = [i for i in range(3,h-3)]
w_valid_range = [i for i in range(3,w-3)]
V = 0
V_pos = [0,0]
for i in range(0,h):
    for j in range(0,w):
        low_i = max(i-3,0)
        up_i = min(i+3,h-1)
        low_j = max(j-3,0)
        up_j = min(j+3,w-1)
        B_r = []
        G_r = []
        R_r = []
        for wi in range(low_i,up_i+1):
            for wj in range(low_j,up_j+1):
                B_r.append(B[wi][wj])
                G_r.append(G[wi][wj])
                R_r.append(R[wi][wj])
        #break
        b_min = np.min(B_r)
        g_min = np.min(G_r)
        r_min = np.min(R_r)
        all_min = min(b_min,g_min,r_min)
        dark_channel[i][j] = all_min
        if all_min > V:
            V = all_min
            V_pos[0] = i
            V_pos[1] = j
cv2.imwrite("dark.jpg",dark_channel)
A_B = B[V_pos[0]][V_pos[1]]
A_G = G[V_pos[0]][V_pos[1]]
A_R = R[V_pos[0]][V_pos[1]]
print(A_B)
print(A_G)
print(A_R)

