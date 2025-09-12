# Used to create pkl files to filter and scale TE data for the simulation
# We used an IC dataset to create pkl files to filter and scale TE data for the simulation
# The same dataset is used to generate the figures given in Section 5.2

import numpy as np
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
pickles_dr = os.path.join(current_dir, "TEpkl")
os.makedirs(pickles_dr, exist_ok=True)

ic_data_dir = os.path.join(current_dir,"ICData")

xnsadd_stddev = np.array([
    0.01,   # xmeasadd[0]
    0.01,   # xmeasadd[1]
    0.01,   # xmeasadd[2]
    0.01,   # xmeasadd[3]
    0.01,   # xmeasadd[4]
    0.125,  # xmeasadd[5]
    0.01,   # xmeasadd[6]
    0.125,  # xmeasadd[7]
    0.01,   # xmeasadd[8]
    0.01,   # xmeasadd[9]
    0.25,   # xmeasadd[10]
    0.1,    # xmeasadd[11]
    0.25,   # xmeasadd[12]
    0.1,    # xmeasadd[13]
    0.25,   # xmeasadd[14]
    0.025,  # xmeasadd[15]
    0.25,   # xmeasadd[16]
    0.1,    # xmeasadd[17]
    0.25,   # xmeasadd[18]
    0.1,    # xmeasadd[19]
    0.25,   # xmeasadd[20]
    0.025,  # xmeasadd[21]
    0.25,   # xmeasadd[22]
    0.1,    # xmeasadd[23]
    0.25,   # xmeasadd[24]
    0.1,    # xmeasadd[25]
    0.25,   # xmeasadd[26]
    0.025,  # xmeasadd[27]
    0.25,   # xmeasadd[28]
    0.1,    # xmeasadd[29]
    0.25,   # xmeasadd[30]
    0.1     # xmeasadd[31]
])

xns_stddev = np.array([
    0.0012,  # xmeas[0]
    18.0,    # xmeas[1]
    22.0,    # xmeas[2]
    0.05,    # xmeas[3]
    0.2,     # xmeas[4]
    0.21,    # xmeas[5]
    0.3,     # xmeas[6]
    0.5,     # xmeas[7]
    0.01,    # xmeas[8]
    0.0017,  # xmeas[9]
    0.01,    # xmeas[10]
    1.0,     # xmeas[11]
    0.3,     # xmeas[12]
    0.125,   # xmeas[13]
    1.0,     # xmeas[14]
    0.3,     # xmeas[15]
    0.115,   # xmeas[16]
    0.01,    # xmeas[17]
    1.15,    # xmeas[18]
    0.2,     # xmeas[19]
    0.01,    # xmeas[20]
    0.01,    # xmeas[21]
    0.25,    # xmeas[22]
    0.1,     # xmeas[23]
    0.25,    # xmeas[24]
    0.1,     # xmeas[25]
    0.25,    # xmeas[26]
    0.025,   # xmeas[27]
    0.25,    # xmeas[28]
    0.1,     # xmeas[29]
    0.25,    # xmeas[30]
    0.1,     # xmeas[31]
    0.25,    # xmeas[32]
    0.025,   # xmeas[33]
    0.05,    # xmeas[34]
    0.05,    # xmeas[35]
    0.01,    # xmeas[36]
    0.01,    # xmeas[37]
    0.01,    # xmeas[38]
    0.5,     # xmeas[39]
    0.5      # xmeas[40]
])

keep_mask = np.array([
    0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 0
])

obs_order = np.hstack([
    np.array([1]),
    xnsadd_stddev,
    xns_stddev,
    np.ones(12)
])

keep_mask_temp = np.hstack([
    np.ones(30),
    np.zeros(9)
]).astype(bool) 

obs_order = obs_order[keep_mask==1]
# print(obs_order.shape)
obs_order = np.tile(obs_order, (10, 1)).reshape(-1)
keep_mask_temp = np.tile(keep_mask_temp,(10, 1)).reshape(-1)

d = np.loadtxt(os.path.join(ic_data_dir,"dataIC.csv") , delimiter=",")
d = d[:20000]/obs_order
d = d[:,keep_mask_temp]

n_warm_up = 5000
n_manifold = 700 + n_warm_up

manifold_data = d[n_warm_up:n_manifold].copy()
max_manifold = np.max(np.abs(manifold_data))

with open(os.path.join(pickles_dr,r"keep_mask.pkl") , "wb") as f:
    pickle.dump(keep_mask_temp, f)
with open(os.path.join(pickles_dr,r"scales.pkl") , "wb") as f:
    pickle.dump(obs_order*max_manifold, f)