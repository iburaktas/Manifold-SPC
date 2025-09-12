import numpy as np
from scipy.stats import rankdata
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path
sys.path.append(current_dir)

import permutation_test

# This is a Python code for Chen's control chart. The control chart runs sequentially.

class DFEWMA:
    def __init__(self, IC_data,  kmax=1000, alpha=0.05, minwin=5, maxwin=10, nbound=10000, lambda_val=0.05,change_point_est = False):
        # Parameters
        self.kmax, self.m0, self.alpha = kmax, IC_data.shape[0], alpha  # # of Phase II data # # of Phase I data # False Alarm Rate
        self.minwin, self.maxwin, self.nbound = minwin, maxwin, nbound  # Min window # Max window  # # of Permutations
        self.lambda_val = lambda_val  # Decay parameter
        self.dim = IC_data.shape[1] # dimension of data
        self.data = np.full((self.m0 + self.kmax, self.dim),  0.0, dtype=np.float64)
        self.data[:self.m0]= IC_data
        self.win = np.clip(np.arange(1, self.kmax + 1), self.minwin, self.maxwin)  # window sizes
        self.sampleNum = np.array(self.m0 + np.arange(1, self.kmax + 1),dtype=np.float64)
        self.change_point = None
        self.change_point_est =change_point_est 
        self.T_v = None

        # Initialize the RankMat, runLength etc
        self.mu, self.sd = self.mu_sd_init()
        self.rankMat = np.full((self.m0 + self.kmax, self.dim),  0.0, dtype=np.float64)
        self.rankMat[:self.m0] = np.apply_along_axis(rankdata, axis=0, arr=self.data[:self.m0])
        self.climit = np.full(self.kmax, 0.0, dtype=np.float64)  # Ensure float type
        self.Tval = np.full(self.kmax, 0.0, dtype=np.float64)  # Ensure float type
        self.runlength = 0

        # Iteration Counters and Checkers
        self.iter_num = 0
        self.flag = True  # if it is true, no alarm

    def iterate(self, new_data):
        if not self.flag:
            print("The control chart cannot iterate anymore due to either the max number of iterations or an alarm")
        else:
            if new_data.shape[0] != 1 or new_data.shape[1] != self.dim:
                print("New observation is not single or the dimensions do not match")
            else:
                self.do_updates(new_data)

    def do_updates(self, new_data):

        cidx = self.m0 + self.iter_num
        eidx = np.arange(0, cidx)

        # Update the rankMat
        for j in range(self.dim):
            largerIdx = np.where(self.data[eidx, j] > new_data[0, j])[0]
            self.rankMat[largerIdx, j] += 1  # Increase rank of larger samples
            self.rankMat[cidx, j] = cidx - len(largerIdx) + 1

        self.data[self.m0 +self.iter_num] = new_data

        # Compute the test statistic
        weight = (1 - self.lambda_val) ** np.arange(self.win[self.iter_num] - 1, -1, -1)
        wm = np.tile(weight[:, None], (1, self.dim))
        current_Tval = np.sum(wm * (self.rankMat[(cidx - self.win[self.iter_num] + 1):cidx + 1, :]-(cidx+2)/2), axis=0)
        current_Tval **= 2
        current_Tval = np.sum(current_Tval)
        current_Tval /= (self.sd[self.iter_num] ** 2)

        current_permT = np.zeros(self.nbound, dtype=np.float64)
        # Compute permutation
        permutation_test.permutationTest(self.rankMat[:cidx + 1, :], 
                                                         self.climit,int(self.m0),
                                                         int(self.rankMat[:cidx + 1, :].shape[0]),
                                                         int(self.rankMat[:cidx + 1, :].shape[1]), 
                                                         int(self.nbound),int(self.minwin), int(self.maxwin),
                                                         current_permT,self.mu,
                                                         self.sd,float(self.lambda_val))         
        # self.lastpermT = current_permT
        #print(current_permT)
        # Update climit, Tval
        # print(len(np.unique(current_permT)))
        # print("iter_num:", self.iter_num)
        # print("climit before assignment:", self.climit[self.iter_num])
        # quantile_value = np.quantile(current_permT, 1 - self.alpha)
        # print("Computed quantile value:", quantile_value)

        self.climit[self.iter_num] = np.quantile(current_permT, 1 - self.alpha)

        # print("climit after assignment:", self.climit[self.iter_num])
        #self.climit[self.iter_num] = np.quantile(current_permT, 1 - self.alpha).copy()
        
        self.Tval[self.iter_num] = current_Tval
        # print(current_Tval)
        # print(self.climit[self.iter_num])
        # Update runlength and iteration number
        if self.Tval[self.iter_num] > self.climit[self.iter_num] or self.iter_num == self.kmax:
            self.runlength = self.iter_num + 1
            self.flag = False
            if self.change_point_est:
                self.change_point = self.estimate_change_point()
        self.iter_num += 1

    def mu_sd_init(self):
        mu = self.win * (self.sampleNum + 1) / 2
        if self.lambda_val == 0:
            sd = (self.win * (self.sampleNum + 1) * (self.sampleNum - self.win)) / 12
        else:
            lam = 1 - self.lambda_val
            temp = lam ** self.win
            sd = ((1 - temp ** 2) * (self.sampleNum + 1) * (self.sampleNum - 1) /
                  ((2 * self.lambda_val - self.lambda_val ** 2) * 12))
            sd -= (((lam - temp) / self.lambda_val ** 2 - (lam ** 2 - temp ** 2) /
                    (2 * self.lambda_val ** 2 - self.lambda_val ** 3)) * (self.sampleNum + 1) / 6)
            sd = np.sqrt(sd)
        return mu.astype(np.float64), sd.astype(np.float64)
    def estimate_change_point(self):

        if self.runlength == 1:
            return 0

        cidx = self.m0 + self.runlength
        # print("n+m0=",cidx)
        win = self.runlength - np.arange(0,self.runlength)
        sd = np.sqrt((win * (cidx + 1) * (cidx - win)) / 12)
        # print("sd:",sd)
        T_v = np.zeros_like(win,dtype=np.float64)


        for iter in range(len(T_v)):
            current_Tval = np.sum(self.rankMat[(cidx - win[iter] ):cidx, :]-(cidx+1)/2, axis=0)
            current_Tval **= 2
            current_Tval = np.sum(current_Tval)
            current_Tval /= (sd[iter] ** 2)
            T_v[iter] = current_Tval
        self.T_v = T_v
        return np.argmax(T_v)
    

class DFUC:
    def __init__(self, IC_data,  kmax=1000, alpha=0.05, minwin=5, maxwin=10, nbound=10000, lambda_val=0.05,change_point_est = False):
        # Parameters
        self.kmax, self.m0, self.alpha = kmax, IC_data.shape[0], alpha  # # of Phase II data # # of Phase I data # False Alarm Rate
        self.minwin, self.maxwin, self.nbound = minwin, maxwin, nbound  # Min window # Max window  # # of Permutations
        self.lambda_val = lambda_val  # Decay parameter
        self.dim = IC_data.shape[1] # dimension of data
        if self.dim != 1:
            raise ValueError("Data must be scalar")
        self.data = np.full((self.m0 + self.kmax, self.dim),  0.0, dtype=np.float64)
        self.data[:self.m0]= IC_data
        self.win = np.clip(np.arange(1, self.kmax + 1), self.minwin, self.maxwin)  # window sizes
        self.sampleNum = np.array(self.m0 + np.arange(1, self.kmax + 1),dtype=np.float64)
        self.change_point = None
        self.change_point_est =change_point_est 
        self.T_v = None

        # Initialize the RankMat, runLength etc
        self.mu, self.sd = self.mu_sd_init()
        self.rankMat = np.full((self.m0 + self.kmax, self.dim),  0.0, dtype=np.float64)
        self.rankMat[:self.m0] = np.apply_along_axis(rankdata, axis=0, arr=self.data[:self.m0])
        self.climit = np.full(self.kmax, 0.0, dtype=np.float64)  # Ensure float type
        self.Tval = np.full(self.kmax, 0.0, dtype=np.float64)  # Ensure float type
        self.runlength = 0

        # Iteration Counters and Checkers
        self.iter_num = 0
        self.flag = True  # if it is true, no alarm

    def iterate(self, new_data):
        if not self.flag:
            print("The control chart cannot iterate anymore due to either the max number of iterations or an alarm")
        else:
            if new_data.shape[0] != 1 or new_data.shape[1] != self.dim:
                print("New observation is not single or the dimensions do not match")
            else:
                self.do_updates(new_data)

    def do_updates(self, new_data):

        cidx = self.m0 + self.iter_num
        eidx = np.arange(0, cidx)

        # Update the rankMat
        for j in range(self.dim):
            largerIdx = np.where(self.data[eidx, j] > new_data[0, j])[0]
            self.rankMat[largerIdx, j] += 1  # Increase rank of larger samples
            self.rankMat[cidx, j] = cidx - len(largerIdx) + 1

        self.data[self.m0 +self.iter_num] = new_data

        # Compute the test statistic
        weight = (1 - self.lambda_val) ** np.arange(self.win[self.iter_num] - 1, -1, -1)
        wm = np.tile(weight[:, None], (1, self.dim))
        current_Tval = np.sum(wm * (np.maximum(0,self.rankMat[(cidx - self.win[self.iter_num] + 1):cidx + 1, :]-(cidx+2)/2)-self.mu[self.iter_num]), axis=0)
        current_Tval **= 1
        current_Tval = np.sum(current_Tval)
        current_Tval /= (self.sd[self.iter_num] ** 1)

        current_permT = np.zeros(self.nbound, dtype=np.float64)
        # Compute permutation
        permutation_test.permutationTestDME(self.rankMat[:cidx + 1, :], 
                                                         self.climit,int(self.m0),
                                                         int(self.rankMat[:cidx + 1, :].shape[0]), 
                                                         int(self.nbound),int(self.minwin), int(self.maxwin),
                                                         current_permT,self.mu,
                                                         self.sd,float(self.lambda_val))         
        # self.lastpermT = current_permT
        #print(current_permT)
        # Update climit, Tval
        # print(len(np.unique(current_permT)))
        # print("iter_num:", self.iter_num)
        # print("climit before assignment:", self.climit[self.iter_num])
        # quantile_value = np.quantile(current_permT, 1 - self.alpha)
        # print("Computed quantile value:", quantile_value)

        self.climit[self.iter_num] = np.quantile(current_permT, 1 - self.alpha)

        # print("climit after assignment:", self.climit[self.iter_num])
        #self.climit[self.iter_num] = np.quantile(current_permT, 1 - self.alpha).copy()
        
        self.Tval[self.iter_num] = current_Tval
        # print(current_Tval)
        # print(self.climit[self.iter_num])
        # Update runlength and iteration number
        if self.Tval[self.iter_num] > self.climit[self.iter_num] or self.iter_num == self.kmax:
            self.runlength = self.iter_num + 1
            self.flag = False
            if self.change_point_est:
                self.change_point = self.estimate_change_point()
        self.iter_num += 1

    def var_zj(self,N):
        if N % 2 == 0:
            return (5*(N**2)-8)/(192*(N**2))
        else:
            return (N**2-1)*(5*N**2+3)/(192*N**4)
        
    def ewma_variance(self,sigma2, N, w, lambda_val):
        lam = 1 - lambda_val
        A = sum(lam ** (2 * (w - j)) for j in range(1, w + 1))  # sum of squares of weights
        S = sum(lam ** (w - j) for j in range(1, w + 1))        # sum of weights
        var = sigma2 * (1 + 1 / (N - 1)) * A - (sigma2 / (N - 1)) * S ** 2
        return var
    
    def mu_sd_init(self):
        mu, sd = self.sampleNum.copy(), self.sampleNum.copy()
        for i in range(len(self.sampleNum)):
            N = self.sampleNum[i]
            if N % 2 == 0:
                mu[i] = 1 / 8
            else:
                mu[i] = (N ** 2 - 1) / (8 * N ** 2)
        
        for i in range(len(self.sampleNum)):
            N = self.sampleNum[i]
            w = self.win[i]
            sigma2 = self.var_zj(N)
            variance = self.ewma_variance(sigma2, N, w, self.lambda_val)
            sd[i] = np.sqrt(variance)

        #print(mu.shape)
        return mu.astype(np.float64), sd.astype(np.float64)
        
    def estimate_change_point(self):

        if self.runlength == 1:
            return 0

        cidx = self.m0 + self.runlength
        # print("n+m0=",cidx)
        win = self.runlength - np.arange(0,self.runlength)
        sd = np.sqrt((win * (cidx + 1) * (cidx - win)) / 12)
        # print("sd:",sd)
        T_v = np.zeros_like(win,dtype=np.float64)


        for iter in range(len(T_v)):
            current_Tval = np.sum(self.rankMat[(cidx - win[iter] ):cidx, :]-(cidx+1)/2, axis=0)
            current_Tval **= 2
            current_Tval = np.sum(current_Tval)
            current_Tval /= (sd[iter] ** 2)
            T_v[iter] = current_Tval
        self.T_v = T_v
        return np.argmax(T_v)
    import numpy as np



    


