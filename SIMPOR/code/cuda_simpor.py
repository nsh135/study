#!/bin/python
import numpy as np
import math as m
from numba import cuda, jit, njit
import time
from tqdm import tqdm
import multiprocessing as mp

from sklearn.neighbors import KernelDensity
import numpy as np
import cupy as cp
from collections import Counter


class SimporCuda:
	"""
	Define a class simpor method that runs on cuda
	"""
	def __init__(self, X, y, h):
		self.X = X
		self.y = y
		self.h = h
		c = Counter(y)
		c = dict(sorted(c.items(), key=lambda item: item[1]))
		self.minor_cls = int(list(c.keys())[0])
		self.major_cls = int(list(c.keys())[1])
		print("Classes of X: {}, minor: {}, major: {}".format(c,self.minor_cls,self.major_cls))
		# SimporCuda.getCudaInfo()
		self.d_X_train_ptr_list = self.transfer_data()	
	
	####------Testing python function---------------------
	def kde_lib(self, x,  kde_ratio=True):
		"""
		Testing function using libraries or self implementation in pure python
		"""
		X, y, minor_cls, major_cls, h = self.X, self.y, self.minor_cls, self.major_cls, self.h; 
		N0 = len(y[y==minor_cls])
		X0 = X[y==minor_cls]
		N1 = len(y[y==major_cls])
		X1 = X[y==major_cls]
		
		selfdefined = False
		#self defined
		if selfdefined:
			log_f0 = SimporCuda.kde_python(X0,x,h)
			log_f1 = SimporCuda.kde_python(X1,x,h)
		else: 	
			##using library
			kde0 = KernelDensity(kernel='gaussian', bandwidth="scott").fit(X0)
			log_f0 = kde0.score_samples(x.reshape(1, -1))[0] # this function return log(density)
			kde1 = KernelDensity(kernel='gaussian', bandwidth="scott").fit(X1)
			log_f1 = kde1.score_samples(x.reshape(1, -1))[0] # this function return log(density)
		if kde_ratio: return log_f0 - log_f1
		else: return log_f0
		
	def kde_python(X, x, h):
		N = len(X)
		d = len(x)
		s=0
		for xi in X:   
			s += (2*np.pi)**(-d/2) * m.exp( -0.5*( np.linalg.norm(x-xi)/h)**2 ) 
		f = s/(N*h**d) 
		return m.log(f)
		
		
	####---------------CUDA related functions--------------------------

	#### Utility function
	def tangent_compute(gradient_array_host,prev_x_array_host,radius_array_host,lr_host ):
		"""
		utility function suport simpor
		"""
		gradient_array = cp.asarray(gradient_array_host)
		prev_x_array = cp.asarray(prev_x_array_host)
		radius_array = cp.asarray(radius_array_host)
		lr = cp.asarray(lr_host)
		
		#project gradient_array vector onto the tangent plane
		dot = cp.expand_dims(cp.sum(gradient_array*prev_x_array,axis=1)/cp.linalg.norm(prev_x_array,axis=1),axis=1)
		tangent_vec_array = gradient_array - (dot**2)*prev_x_array #(N,dims)
		#normalize tangent vector 
		tangent_vec_array /= cp.expand_dims(cp.linalg.norm(tangent_vec_array,axis=1),axis=1)   # still on new origin coordinate
		prev_x_array_normalized = prev_x_array / cp.expand_dims(cp.linalg.norm(prev_x_array,axis=1),axis=1)  
		###update x
		phi = lr*np.pi #generate phi
		cur_x_array = (prev_x_array_normalized * cp.cos(phi) + tangent_vec_array*cp.sin(phi))*radius_array  #Grad ascent on the origin coordinate
		return 	cp.asnumpy(tangent_vec_array), cp.asnumpy(prev_x_array_normalized), cp.asnumpy(cur_x_array)

	### Simpor cuda 
	def getCudaInfo():
		gpu = cuda.get_current_device()
		print("name = %s" % gpu.name)
		print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
		print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
		print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
		print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
		print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
		print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
		print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
		print("maxSharedMemoryPerBlock = %s" % 
		str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
		print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
		print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
		print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
		print("warpSize = %s" % str(gpu.WARP_SIZE))
		print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
		print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
		print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))
	# return 2 lists of pointers according to classes
	def transfer_data(self):
		"""
		Transfer each data class to cuda
		"""
# 		t = time.time()
		d_X_train = []
		for cls in range(2):
			X = self.X[self.y==cls]
			d_X_train.append(cuda.to_device(X))
# 		print("Transfering Data to Cuda TIME: {:0.3f} s".format(time.time()-t ))
		return d_X_train

	# each thread process 1 point in X_train in the given class
	@cuda.jit
	def f_kernel1(x, d_X_train_cls, h,d_output_array_2D, N_samples):
		"""
		kernel to estimate log likelyhood for each point in x
		Params
		--------
		x :points need to be estimated array(samples, dim)
		d_X_train_cls : X_train by class, only 1 class at once
		h: bandwidth of Gaussian kernel
		"""
		tx, ty = cuda.grid(2)
		if tx >= N_samples or ty >= x.shape[0]:
			return

		#Each thread proccess for one sample in training data
		Xi = d_X_train_cls[tx]  #go through training samples
		d = x.shape[1]  #sample dim
		#Compute for each esimated points
		xj = x[ty]
		norm_sqr = 0
		for k in range(d): #go through diemntions of each point to compute L2 norm
			norm_sqr += (xj[k] - Xi[k] )**2
		norm = m.sqrt(norm_sqr)
		d_output_array_2D[ty, tx] =  m.exp( -0.5*(  (norm/h)**2  ) ) / (N_samples*h**d)


	def calculate_f1(x, d_X_train_ptr_cls, h, N_samples):
		""" 
		Estimate kde(s) for x(s) 
		x: example x 
		d_X_train_cls:  data pointer given class of sample x
		h: bandwidth of kernel
		N_samples : number of samples in class of x
		"""
		N_estimate_samples = x.shape[0]
		# allocate output array
		out2D = np.zeros(shape = (N_estimate_samples, N_samples))
		d_output_array_2D = cuda.to_device(out2D)

		d = x.shape[1] #sample dimention
		#estimate log-likelyhood for more than 1 points
		threadsperblock = (8, 8)
		blockspergrid_x = m.ceil(N_samples / threadsperblock[0])
		blockspergrid_y = m.ceil(N_estimate_samples / threadsperblock[1])
		blockspergrid = (blockspergrid_x, blockspergrid_y)
		total = blockspergrid_x*threadsperblock[0]*blockspergrid_y*threadsperblock[1]
		if False:
			print("Initiate CUDA kernel for\
			\nthreadsperblock: {}\
			\nblockspergrid: {}\
			\nThreads Total: {}\n-----".format(threadsperblock,blockspergrid,total))

		# calling kernel
		SimporCuda.f_kernel1[blockspergrid, threadsperblock](x, d_X_train_ptr_cls, h, d_output_array_2D, N_samples)

		# compute sum for each estimated point
		# cp_sum_reduce()
		cp_sum_reduce = cp.ReductionKernel(
			'T x',  # input params
			'T y',  # output params
			'x',  # map
			'a + b',  # reduce
			'y = a',  # post-reduction map
			'0',  # identity value
			'cp_sum_reduce'  # kernel name
			)
		fs = cp_sum_reduce(d_output_array_2D, axis=1)
		return cp.asnumpy(fs)

	@cuda.jit
	def f_kernel2(x, d_X_train_cls_major, d_X_train_cls_minor , h,d_output_array_2D, N_CLASS_major,N_CLASS_minor ):
		"""
		kernel to estimate log likelyhood for each point in x
		Params
		--------
		x :points need to be estimated array(samples, dim)
		d_X_train_cls : X_train by class, only 1 class at once
		h: bandwidth of Gaussian kernel
		-----
		arrays in cuda
								Trianing Data Array
					|d_X_train_cls_major |d_X_train_cls_minor|
		Results		|					 |					 |
		(Nsamples)	|					 |					 |
					|					 |					 | 
		"""
		N_samples =  N_CLASS_major + N_CLASS_minor
		tx, ty = cuda.grid(2)
		if tx >= N_samples or ty >= x.shape[0]:
			return
		elif tx < N_CLASS_major:
			#Each thread proccess for one sample in training data
			Xi = d_X_train_cls_major[tx]  #go through training samples
			N = N_CLASS_major
		elif tx >= N_CLASS_major and tx < N_samples:  # Concatenate these 2 arrays |d_X_train_cls_major |d_X_train_cls_minor|
			Xi = d_X_train_cls_minor[tx- N_CLASS_major ]
			N = N_CLASS_minor

		d = x.shape[1]  #sample dim
		#Compute for each esimated points
		xj = x[ty]
		norm_sqr = 0
		for k in range(d): #go through diemntions of each point to compute L2 norm
			norm_sqr += (xj[k] - Xi[k] )**2
		norm = m.sqrt(norm_sqr)
		d_output_array_2D[ty, tx] =  m.exp( -0.5*(  (norm/h)**2  ) ) / (N*h**d)


	def calculate_f2(x, d_X_train_ptr_cls_major, d_X_train_ptr_cls_minor, h, N_CLASS_major,N_CLASS_minor ):
		""" 
		Estimate kde(s) for x(s) 
		x: estmated samples x 
		d_X_train_cls:  data pointer given class of sample x
		h: bandwidth of kernel
		N_samples : number of samples in both calsses
		"""
		N_samples =  N_CLASS_major + N_CLASS_minor
		N_estimate_samples = x.shape[0]
		# allocate output array
		out2D = np.zeros(shape = (N_estimate_samples, N_samples))
		d_output_array_2D = cuda.to_device(out2D)
		d = x.shape[1] #sample dimention
		#estimate log-likelyhood for more than 1 points
		threadsperblock = (8, 8)
		blockspergrid_x = m.ceil(N_samples / threadsperblock[0])
		blockspergrid_y = m.ceil(N_estimate_samples / threadsperblock[1])
		blockspergrid = (blockspergrid_x, blockspergrid_y)
		total = blockspergrid_x*threadsperblock[0]*blockspergrid_y*threadsperblock[1]
		# calling kernel
		SimporCuda.f_kernel2[blockspergrid, threadsperblock](x, d_X_train_ptr_cls_major, d_X_train_ptr_cls_minor,\
			 h, d_output_array_2D, N_CLASS_major,N_CLASS_minor )
		# compute sum for each estimated point
		# cp_sum_reduce()
		cp_sum_reduce = cp.ReductionKernel(
			'T x',  # input params
			'T y',  # output params
			'x',  # map
			'a + b',  # reduce
			'y = a',  # post-reduction map
			'0',  # identity value
			'cp_sum_reduce'  # kernel name
			)
		f_major = cp_sum_reduce(d_output_array_2D[:,0:N_CLASS_major], axis=1)
		f_minor = cp_sum_reduce(d_output_array_2D[:,N_CLASS_major:N_samples], axis=1)
		return f_major, f_minor

	def Ratio_kde_CUDA(self, x, CUDA_device= 0):
		
		# cuda.select_device(CUDA_device) -> Not Supported yet 
		# t = time.time()
		X, y, minor_cls, major_cls, h = self.X, self.y, self.minor_cls, self.major_cls, self.h; 
		d_X_train_ptr_list = self.d_X_train_ptr_list
		N_CLASS_minor = len(y[y==minor_cls])
		N_CLASS_major = len(y[y==major_cls])

		use_kernel1 = False
		if use_kernel1:
			f_minor = SimporCuda.calculate_f1(x, d_X_train_ptr_list[minor_cls], h, N_CLASS_minor)
			f_major = SimporCuda.calculate_f1(x, d_X_train_ptr_list[major_cls], h, N_CLASS_major)	
		else:
			f_major, f_minor = SimporCuda.calculate_f2(x, d_X_train_ptr_list[major_cls], d_X_train_ptr_list[minor_cls], h, N_CLASS_major,N_CLASS_minor )
		
		posterior_ratios = f_minor/f_major
		log_kde_ratio = cp.log(posterior_ratios)
		# print("Finish ratio kde computing with CUDA in {:.2f} seconds".format( (time.time()-t)) )
		return cp.asnumpy(log_kde_ratio) #np array

	def self_check(self,x):
		"""
		self sanity check cuda computation
		x: points need to be estimated
		"""
		output2=[]
		##test for each centroid point
		# Test on CUDA
		time0 = time.time()
		output1 = self.Ratio_kde_CUDA(x,  CUDA_device=0)      
		print("time CUDA :", time.time()-time0)

		# # Test on CPU
		time1 = time.time()
		for xi in x:
			output2.append(self.kde_lib(xi ) )
		output2=np.array(output2)
		print("time cpu :", time.time()-time1)
		# print("Ratio_kde_CUDA output ", output1, output2) ## for sanity check, they must be equal
		check = np.allclose(output1,output2)
		assert check, "Error: Ratio_kde_CUDA didn't match pure python: {} v.s {}".format(output1,output2)
		return check

def sanity_check():
	N_CLASS1 = 1000
	N_CLASS2 = 1000
	DIM = 30
	X_train = np.random.rand(N_CLASS1+N_CLASS2, DIM)
	y_train = np.array([0]* N_CLASS1 + [1]*N_CLASS2)
	print("Sanity Chek Cuda computation..")
	SimporCuda.getCudaInfo()
	print("X.shape: ",X_train.shape)
	x = np.random.rand(10, DIM)
	x = np.array(x,dtype = 'float32')
	print("x.sahpe", x.shape)
	h=0.2
	sc = SimporCuda(X_train, y_train, h)
	sc.self_check(x)

if __name__ == "__main__":
	sanity_check()

		 


		

