import wfdb
import math
import numpy as np
from scipy.linalg import hankel
from scipy.linalg import svd
import matplotlib.pyplot as plt
import time


start = time.time()

signal, fields = wfdb.rdsamp('s3_run', channels=[1,8,9,10]) #load signal
print(fields)

ppg=signal[:,0]      #channel split
accel_x=signal[:,1]
accel_y=signal[:,2]
accel_z=signal[:,3]


N=400   # N+L-1 = 2048
L=1649
indices = [i for i, x in enumerate(ppg) if math.isnan(x)]   #deleting NaN from the end of the signal
ppg_nonan = np.delete(ppg, indices)
accel_x_nonan = np.delete(accel_x, indices)
accel_y_nonan = np.delete(accel_y, indices)
accel_z_nonan = np.delete(accel_z, indices)

sample_divider = 2048 #window length
sample_step = 512 #sample shift


n=0
while n<(len(ppg_nonan)//sample_step):
    p=ppg_nonan[n*512:sample_divider+n*sample_step]
    x, y, z = accel_x_nonan[n*sample_step:sample_divider+n*sample_step], accel_y_nonan[n*sample_step:sample_divider+n*sample_step], accel_z_nonan[n*sample_step:sample_divider+n*sample_step]
    p=p.T
    x=x.T
    y=y.T
    z=z.T
    h_ppg = hankel(p)
    h_x, h_y, h_z = hankel(x), hankel(y), hankel(z)
    h_ppg = h_ppg[:N, :L]
    h_x, h_y, h_z = h_x[:N,:L], h_y[:N,:L], h_z[:N,:L]
    h_xyz = np.concatenate((h_x,h_y,h_z),axis=1)

    S, Lambda, RT = svd(h_xyz)
    vect_norm = np.linalg.norm(Lambda)

    Lambda_a=Lambda/vect_norm
    '''
    plt.plot(Lambda_a[:20])
    plt.show()   #ploting the lambda vectors
    '''
    S, Lambda, RT = svd(h_ppg)
    vect_norm = np.linalg.norm(Lambda)
    Lambda_p=Lambda/vect_norm

    if n==0:
        vectors = np.append(Lambda_a,Lambda_p)
    else:
        r=np.append(Lambda_a,Lambda_p)
        vectors= np.vstack([vectors,r])
    print(n)
    n+=1

end = time.time()
print(end - start)
np.savetxt("s3_run_high_rng_accel.csv", vectors, delimiter=";")