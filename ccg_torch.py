# %%
import os
import pickle
import itertools
import numpy as np
from scipy import sparse
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
from joblib import Parallel, delayed
import time
import torch
cpu_num = 8 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
# %%
def save_npz(matrix, filename):
    matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
    sparse_matrix = sparse.csc_matrix(matrix_2d)
    np.savez(filename, [sparse_matrix, matrix.shape])
    return 'npz file saved'

def load_npz(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True)
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d
    # new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

def load_npz_3d(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True)
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

def save_sparse_npz(matrix, filename):
	matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
	sparse_matrix = sparse.csc_matrix(matrix_2d)
	with open(filename, 'wb') as outfile:
		pickle.dump([sparse_matrix, matrix.shape], outfile, pickle.HIGHEST_PROTOCOL)

def load_sparse_npz(filename):
	with open(filename, 'rb') as infile:
		[sparse_matrix, shape] = pickle.load(infile)
		matrix_2d = sparse_matrix.toarray()
	return matrix_2d.reshape(shape)

class pattern_jitter():
		def __init__(self, num_sample, sequences, L, R=None, memory=True):
				self.num_sample = num_sample
				self.sequences = np.array(sequences)
				if len(self.sequences.shape) > 1:
						self.N, self.T = self.sequences.shape
				else:
						self.T = len(self.sequences)
						self.N = None
				self.L = L
				self.memory = memory
				if self.memory:
						assert R is not None, 'R needs to be given if memory is True!'
						self.R = R
				else:
						self.R = None

		def spike_timing2train(self, spikeTrain):
				if len(spikeTrain.shape) == 1:
						spikeData = np.zeros(self.T)
						spikeData[spikeTrain.astype(int)] = 1
				else:
						spikeData = np.zeros((spikeTrain.shape[0], self.T))
						spikeData[np.repeat(np.arange(spikeTrain.shape[0]), spikeTrain.shape[1]), spikeTrain.ravel().astype(int)] = 1
				return spikeData

		def getSpikeTrain(self, spikeData):
				if len(spikeData.shape) == 1:
						spikeTrain = np.squeeze(np.where(spikeData>0)).ravel()
				else:
						spikeTrain = np.zeros((spikeData.shape[0], len(np.where(spikeData[0, :]>0)[0])))
						for i in range(spikeData.shape[0]):
								spikeTrain[i, :] = np.squeeze(np.where(spikeData[i, :]>0)).ravel()
				return spikeTrain

		def getInitDist(self):
				initDist = np.random.rand(self.L)
				return initDist/initDist.sum()

		def getTransitionMatrices(self, num_spike):
				tDistMatrices = np.zeros((num_spike - 1, self.L, self.L))
				for i in range(tDistMatrices.shape[0]):
						matrix = np.random.rand(self.L, self.L)
						stochMatrix = matrix/matrix.sum(axis=1)[:,None]
						tDistMatrices[i, :, :] = stochMatrix.astype('f')
				return tDistMatrices

		def getX1(self, jitter_window, initDist):
				
				randX = np.random.random()
				ind = np.where(randX <= np.cumsum(initDist))[0][0]
				return jitter_window[0][ind]

		def initializeX(self, initX, Prob):
				return initX + np.sum(Prob == 0)

		def getOmega(self, spikeTrain):
				Omega = []
				n = spikeTrain.size
				for i in range(n):
						temp = spikeTrain[i] - np.ceil(self.L/2) + 1
						temp = max(0, temp)
						temp = min(temp, self.T - self.L)
						Omega.append(np.arange(temp, temp + self.L, 1))
				return Omega

		# def getOmega(self, spikeTrain):
		#     Omega = []
		#     n = spikeTrain.size
		#     for i in range(n):
		#         temp = spikeTrain[i] - np.ceil(self.L/2) + 1
		#         lower_bound = max(0, temp)
		#         upper_bound = min(temp + self.L, self.T)
		#         Omega.append(np.arange(lower_bound, upper_bound, 1))
		#     return Omega

		def getGamma(self, spikeTrain):
				Gamma = []
				ks = [] # list of k_d
				ks.append(0)
				n = spikeTrain.size
				temp = int(spikeTrain[ks[-1]]/self.L)*self.L
				temp = max(0, temp)
				temp = min(temp, self.T - self.L)
				Gamma.append(np.arange(temp, temp + self.L, 1))
				for i in range(1, n):
						if spikeTrain[i] - spikeTrain[i-1] > self.R:
								ks.append(i)
						temp = int(spikeTrain[ks[-1]]/self.L)*self.L+spikeTrain[i]-spikeTrain[ks[-1]]
						temp = max(0, temp)
						temp = min(temp, self.T - self.L)
						Gamma.append(np.arange(temp, temp + self.L, 1))
				return Gamma

		def getSurrogate(self, spikeTrain, initDist, tDistMatrices):
				surrogate = []
				if self.memory:
						jitter_window = self.getGamma(spikeTrain)
				else:
						jitter_window = self.getOmega(spikeTrain)
				givenX = self.getX1(jitter_window, initDist)
				surrogate.append(givenX)
				for i, row in enumerate(tDistMatrices):
						if self.memory and spikeTrain[i+1] - spikeTrain[i] <= self.R:
								givenX = surrogate[-1] + spikeTrain[i+1] - spikeTrain[i]
						else:
								index = np.where(np.array(jitter_window[i]) == givenX)[0]
								p_i = np.squeeze(np.array(row[index]))
								initX = self.initializeX(jitter_window[i + 1][0], p_i)
								randX = np.random.random()
								# safe way to find the ind
								larger = np.where(randX <= np.cumsum(p_i))[0]
								if larger.shape[0]:
										ind = larger[0]
								else:
										ind = len(p_i) - 1
								givenX = initX + np.sum(p_i[:ind]!=0)
						givenX = min(self.T - 1, givenX) # possibly same location
						if givenX in surrogate:
								locs = jitter_window[i + 1]
								available_locs = [loc for loc in locs if loc not in surrogate]
								givenX = np.random.choice(available_locs)
						surrogate.append(givenX)
				return surrogate

		def sample_spiketrain(self, spikeTrain, initDist, tDistMatrices):
				spikeTrainMat = np.zeros((self.num_sample, spikeTrain.size))
				for i in tqdm(range(self.num_sample), disable=True):
						surrogate = self.getSurrogate(spikeTrain, initDist, tDistMatrices)
						spikeTrainMat[i, :] = surrogate
				return spikeTrainMat

		def jitter(self):
				# num_sample x N x T
				if self.N is not None:
						jittered_seq = np.zeros((self.num_sample, self.N, self.T))
						for n in range(self.N):
								spikeTrain = self.getSpikeTrain(self.sequences[n, :])
								num_spike = spikeTrain.size
								if num_spike:
										initDist = self.getInitDist()
										tDistMatrices = self.getTransitionMatrices(num_spike)
										sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
										jittered_seq[:, n, :] = self.spike_timing2train(sampled_spiketrain)
								else:
										jittered_seq[:, n, :] = np.zeros((self.num_sample, self.T))
				else:
						spikeTrain = self.getSpikeTrain(self.sequences)
						num_spike = spikeTrain.size
						initDist = self.getInitDist()
						tDistMatrices = self.getTransitionMatrices(num_spike)
						sampled_spiketrain = self.sample_spiketrain(spikeTrain, initDist, tDistMatrices)
						jittered_seq = self.spike_timing2train(sampled_spiketrain).squeeze()
				return jittered_seq



### Older version SLow

def ccg_pair(mat1, mat2, firing_rates, row_a, row_b, M, window=100):
	# mat1 is padded on both sides while mat2 is padded only on the right
	if firing_rates[row_a] * firing_rates[row_b] > 0: # there could be no spike in a certain trial
		px, py = mat1[row_a, :], mat2[row_b, :]
		T = as_strided(px[window:], shape=(window+1, M + window),
				strides=(-px.strides[0], px.strides[0])) # must be py[window:], why???????????
		return (T @ py) / ((M-np.arange(window+1))/1000 * np.sqrt(firing_rates[row_a] * firing_rates[row_b]))
	else:
		return np.zeros(window+1)


### New version, fft, cuda accerleration

def ccg_pair_torch(mat1, mat2, firing_rates, M, window=100,device=torch.device('cuda:0'),double_side=False):
	# mat1 is padded on both sides while mat2 is padded only on the right
	# print("new method")
	N = mat1.shape[0]
	mat1 = torch.tensor(mat1[:,window:]).to(device)
	mat2 = torch.tensor(mat2).to(device)
	firing_rates = torch.tensor(firing_rates).to(device)
	mat1_complex = torch.view_as_complex(torch.stack([mat1, torch.zeros_like(mat1)], dim=-1)).to(device)
	mat2_complex = torch.view_as_complex(torch.stack([mat2, torch.zeros_like(mat2)], dim=-1)).to(device)
	# Compute FF2
	mat1_complex_rep = torch.repeat_interleave(mat1_complex, torch.tensor([N],device=device), dim=0)
	mat2_complex_tile = mat2_complex.tile((N,1))
	X = torch.fft.fft(mat1_complex_rep)
	Y = torch.fft.fft(mat2_complex_tile)
	cross_corr_freq = X * torch.conj(Y)
	cross_corr_time = torch.fft.ifft(cross_corr_freq)
	cross_corr_time = cross_corr_time.real
	
	fr_rep = torch.repeat_interleave(firing_rates, torch.tensor([N],device=device), dim=0)
	fr_tile = firing_rates.tile((N,))

	if double_side:
		cross_corr_time = torch.roll(cross_corr_time, shifts=window, dims=1)
		cross_corr = cross_corr_time[:,:(2*window+1)]
		tri_angle_r = (torch.arange(M-1,M-window-1,step=-1)/1000)
		tri_angle_l = (torch.arange(M-window,M+1)/1000)
		tri_angle = torch.cat((tri_angle_l,tri_angle_r)).to(device)
	else:
		cross_corr_time = torch.roll(cross_corr_time, shifts=mat1.shape[1]-1, dims=1)
		cross_corr = cross_corr_time[:,-(window+1):]
		tri_angle = (torch.arange(M-window,M+1)/1000).to(device) #flip the triangle function
	# tri_angle = ((M-torch.arange(window+1))/1000).to(device)
	
	tri_angle = tri_angle.repeat([N*N,1])
	# print(f"{tri_angle.shape},{fr_rep.repeat([cross_corr.shape[1],1]).T.shape},{fr_tile.repeat([cross_corr.shape[1],1]).T.shape}")
	# cross_corr = cross_corr/tri_angle
	cross_corr = cross_corr / (tri_angle * torch.sqrt(fr_rep.repeat([cross_corr.shape[1],1]).T * fr_tile.repeat([cross_corr.shape[1],1]).T))
	return np.flip(cross_corr.reshape(N,N,-1).cpu().numpy(),axis=-1)


def get_all_ccg_parallel(matrix, window=100, disable=True, num_cores=24,double_side=False,device=torch.device('cuda:0')): ### parallel CCG
	N, M =matrix.shape # neuron, time
	if double_side:
		ccg=np.empty((N,N,2*window+1))
	else:
		ccg=np.empty((N,N,window+1))
	ccg[:] = np.nan
	firing_rates = np.count_nonzero(matrix, axis=1) / (matrix.shape[1]/1000) # Hz instead of kHz
	### current spike of neuron A is compared with future spike of neuron B
	### should be directed correlation A -> B
	norm_mata = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1)
	norm_matb = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1)
	ccg = ccg_pair_torch(norm_mata,norm_matb,firing_rates,M,double_side = double_side,device=device)
	ccg[np.isnan(ccg)]=0
	for i in range(ccg.shape[0]):
		ccg[i,i,:]=0
	return ccg

# def get_all_ccg_parallel(matrix, window=100, disable=True, num_cores=24): ### parallel CCG
# 	N, M =matrix.shape # neuron, time
# 	ccg=np.empty((N,N,window+1))
# 	ccg[:] = np.nan
# 	firing_rates = np.count_nonzero(matrix, axis=1) / (matrix.shape[1]/1000) # Hz instead of kHz
# 	### current spike of neuron A is compared with future spike of neuron B
# 	### should be directed correlation A -> B
# 	norm_mata = np.concatenate((np.zeros((N, window)), matrix.conj(), np.zeros((N, window))), axis=1)
# 	norm_matb = np.concatenate((matrix.conj(), np.zeros((N, window))), axis=1) # must concat zeros to the left, why???????????
# 	#####
# 	total_list = list(itertools.permutations(range(N), 2))
# 	result = Parallel(n_jobs=num_cores)(delayed(ccg_pair)(norm_mata, norm_matb, firing_rates, row_a, row_b, M, window) for row_a, row_b in tqdm(itertools.permutations(range(N), 2), total=len(total_list), miniters=int(len(total_list)/50), maxinterval=200, disable=disable)) #
# 	for i in range(len(total_list)):
# 		row_a, row_b = total_list[i]
# 		ccg[row_a, row_b, :] = result[i]
# 	return ccg


def save_mean_ccg_corrected_parallel(sequences, fname, num_jitter=10, L=25, window=100, disable=True,double_side=False,device=torch.device('cuda:0')): 
	num_neuron, num_trial, _ = sequences.shape
	# num_trial = min(num_trial, 1000) # at most 1000 trials
	if double_side:
		ccg, ccg_jittered = np.zeros((num_neuron, num_neuron, 2*window + 1)),np.zeros((num_neuron, num_neuron, 2*window + 1))
	else:
		ccg, ccg_jittered = np.zeros((num_neuron, num_neuron, window + 1)),np.zeros((num_neuron, num_neuron, window + 1))
	pj = pattern_jitter(num_sample=num_jitter, sequences=sequences[:,0,:], L=L, memory=False)
	for m in tqdm(range(num_trial), disable=False):
		# print('Trial {} / {}'.format(m+1, num_trial))
		ccg += get_all_ccg_parallel(sequences[:,m,:], window, disable=disable,double_side=double_side,device=device) # N x N x window  ###time consuming
		pj.sequences = sequences[:,m,:]
		sampled_matrix = pj.jitter() # num_sample x N x T
		for i in range(num_jitter):
			ccg_jittered += get_all_ccg_parallel(sampled_matrix[i, :, :], window, disable=disable,double_side=double_side,device=device)  ###time consuming
	ccg = ccg / num_trial
	ccg_jittered = ccg_jittered / (num_jitter * num_trial)
	# for n in range(num_neuron):
	# 	ccg[n,n,:] = 0
	# 	ccg_jittered[n,n,:] = 0
	save_sparse_npz(ccg, fname)
	save_sparse_npz(ccg_jittered, fname.replace('ccg_', 'ccg_jittered_'))
