import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
import math as m
from tqdm.notebook import tqdm
from collections import Counter
import copy 
# from cuda_test import kde_CUDA
from sklearn.neighbors import KernelDensity as sklearn_kde
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import time

CUDA_device = 0
onetime_plot = True #plot entropy histogram for debugging

def get_unlabeled_idx(X_train, labeled_idx):
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]
    
def uncertainty_sampling( X_train, threshold,model):
    """
    Obtain informative samples based on high entropy samples. This function implement one time training proccess. 
    Prameters:
    X_train: entire training set
    threshold: entropy threshold - Informative portion (threshold = 0.1 will take 10% of highest entropy samples)
    return:
    selected_indices: indices of samples with the least confidence (high entropy) 
    """
    size = len(X_train)
    predictions = model.predict_proba(X_train)
    predictions_entropy = entropy(predictions, axis=1, base=X_train.shape[1] )
    # selected_indices =  [i for i,v in enumerate(predictions_entropy) if v>threshold] 
    
    #take high entropy portion of predictions 
    selected_indices = np.argsort(predictions_entropy)[-int(size*threshold):]
    

    
    return selected_indices



def sample_spherical(centroid_x ,radius, npoints):
    """
    sample a point on n-sphere centered by centroid_x
    Unit circle
    """
    #generate points in n-sphere centered by the origin
    ndim = len(centroid_x)
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    vec = vec.T
    
    #sanity check if the length of points are 1
    for v in vec:# loop all points
        s = 0
        for i in range(ndim):# loop all dims
            s += v[i]**2
        if s - radius**2 > 0.0000001 : raise ValueError('Initial value is not on sphere')
            
    # shift to be centered by centroid_x
    vec = vec + centroid_x
    return vec

def kde(x, X, y, minor_cls=0,major_cls=1, h=0.3, kde_ratio=True):
    """
    Compute of log posterior ratio (log f=f0/f1) 
    X,y :  entire dataset
    minor_cls: class we want to generate synthetic data, should be the numerator
    major_cls: majority class
    h: Gaussian kernel bandwidth
    ratio: whether using ratio for informative region, or f0 for the remainder
    
    if self-defined
    -------------------
     s0 = 0
     for xi in X0:
         s0 += m.exp( -0.5*( np.linalg.norm(x-xi)**2 / h**2  ) )
     f0 = s0/(N0+1e-5)
  
     s1 = 0
     for xi in X1:
         s1 += m.exp( -0.5*( np.linalg.norm(x-xi)**2 / h**2  ) )
     f1 = s1/(N1+1e-5)
     return m.log(f0+1e-5) - m.log(f1+1e-5)
    """
    N0 = len(y[y==minor_cls])
    X0 = X[y==minor_cls]
    N1 = len(y[y==major_cls])
    X1 = X[y==major_cls]
    
    ##using library
    kde0 = sklearn_kde(kernel='gaussian', bandwidth="scott").fit(X0)
    log_f0 = kde0.score_samples(x.reshape(1, -1))[0] # this function return log(density)
    kde1 = sklearn_kde(kernel='gaussian', bandwidth="scott").fit(X1)
    log_f1 = kde1.score_samples(x.reshape(1, -1))[0] # this function return log(density)
    if kde_ratio: return log_f0 - log_f1
    else: return log_f0
    
    
def derivative_f(x,fun, CUDA=False): 
    """
    Compute derivative of a function at one sample or samples(CUDA)
    fun : input function
    x: value of x where want to compute derivative
    ---
    return 
    Particial derivatives along dimensions
    """
    delta = 1e-5
    if not CUDA:
        D = x.shape[0] # number of dimensions of x
        df = [] # store partial dirivative
        
        #compute partial derivative for each dimention of x
        for i in range(D): # i run thru all dimension
            x_plus_delta = copy.deepcopy(x)
            x_minus_delta = copy.deepcopy(x)
            x_plus_delta[i] = x_plus_delta[i] + delta
            x_minus_delta[i] = x_minus_delta[i] -  delta
            df.append( (fun(x_plus_delta)-fun(x_minus_delta))/(2*delta) )
        return np.array(df)
    else: #Cuda mode, x becomes array of samples (N_samples,dimension)
        D= x.shape[1]
        df_array = []
        #t = time.time()

        #compute partial derivative over each dimension for all points at once
        x_plus_delta = cp.array(x)
        x_minus_delta = cp.array(x)
        for i in range(D): #x (N_samples,dimensions)
            if i != 0: # return to original x from previous computation
                x_plus_delta[:,i-1] = x_plus_delta[:,i-1] - delta
                x_minus_delta[:,i-1] = x_minus_delta[:,i-1] +  delta
            x_plus_delta[:,i] = x_plus_delta[:,i] + delta
            x_minus_delta[:,i] = x_minus_delta[:,i] -  delta
            #differential derivative at the dimension i'th 
            df_ith = (fun(x_plus_delta)-fun(x_minus_delta))/(2*delta) 
            df_array.append(df_ith )
        #print("Time to take derivative:",time.time()-t)
        return np.array(df_array).T #(N_samples,df_dimension)




def gd( X, y, minor_cls, major_cls, h, radius, x,  max_iters=100, lr=0.01, tolerance=0.0000001,GD_patience=10, kde_ratio=True, verbose=0, CUDA= False ):
    """
    Implement Gradient Ascent on a funcion "fun(x)" at x
    X,y : considering Dataset 
    minor_cls, major_cls: minority and majority classes 
    h: bandwidth of the Gaussian kernel 
    radius: radius of n-sphere
    x: the selected minority example x that synthetic data generated from
    max_iters: maximum number of gradient iterations 
    lr: gradient learning rate
    radius: the raius of n-sphere surrounding example x (should be small )
    tolerance: determine step size can be consider as a increasement 
    return: value of the optima 
    
    *note: if CUDA: x, radius are arrays instead of scalars 
    """
    
    def Ratio_Kde(x):
        """
        This is the ratio function we want to minimize
        -----
        return: posterior ratio 
        """
        if CUDA:
            return SC.Ratio_kde_CUDA(x, CUDA_device =0)
        else:
            return kde(x, X, y, minor_cls=minor_cls, major_cls=major_cls, h=h, kde_ratio = kde_ratio)

    if not CUDA:
        radius = radius
        new_origin = x     # shift the coordicate to a new origin (x)
        #find an initial point (int_x) that lay on n-sphere centered by the example x
        init_x = sample_spherical(new_origin,radius, 1)[0]           
        
        cur_x = init_x - new_origin # The algorithm starts at init_x on new origin
        tolerance = tolerance #This tells us when to stop the algorithm
        previous_step_size = 1 #
        iters = 0 #iteration counter
        df = lambda at_x: derivative_f(at_x,Ratio_Kde, CUDA) #Gradient of our function at cur_x
        
        patience = 0
        old = Ratio_Kde(cur_x)
        while  iters < max_iters and patience<GD_patience:
            if previous_step_size < tolerance:
                patience += 1
            prev_x = cur_x #Store current x value in prev_x
            
            # compute gradient vector at prev_x
            gradient = df(prev_x+new_origin)  # grantident computed based on the origin coordinate
            
            #project gradient vector onto the tangent plane
            tangent_vec = gradient - (np.dot(gradient,prev_x)/norm(prev_x)**2)*(prev_x)
            
            #normalize tangent vector 
            tangent_vec /= norm(tangent_vec)   # still on new origin coordinate
            
            prev_x_normalized = prev_x / norm(prev_x)
            
            ###update x
            phi = lr*np.pi #generate phi
            cur_x = (prev_x_normalized * np.cos(phi) + tangent_vec*np.sin(phi))*radius  #Grad ascent on the origin coordinate
            previous_step_size = abs(Ratio_Kde(cur_x+new_origin)- Ratio_Kde(prev_x+new_origin))   #Change in target function  
            # assert abs(norm(cur_x)-radius)<0.00001, "cur_x {} is not equal to radius {}".format(norm(cur_x), radius)    

            if verbose==1: print("\nIter: "+ str(iters) +"   Loss : ", Ratio_Kde(cur_x),  "cur_x: ",cur_x)
            iters = iters+1 #iteration count
        new = Ratio_Kde(cur_x)
        if verbose: print("\nThe local minimum occurs at", cur_x + new_origin)
        if verbose: print("Old RatioKDE:{}, new RatioKDE: {}  RatioKDE Difference(old-new):{}", old, new , old-new)
        return cur_x + new_origin ## return x on original origin coordinate

    else: #CUDA mode, process all points at once
        radius_array = radius
        new_origin_array = x    # shift the coordicate to a new origin (x)
        #find an initial point (int_x) that lay on n-sphere centered by the example x
        init_x_array = np.array( [ sample_spherical(no,rad, 1)[0] for (no,rad) in zip(new_origin_array,radius_array)])
        
        cur_x_array = init_x_array - new_origin_array # The algorithm starts at init_x on new origin
        iters = 0 #iteration counter

        df_array = lambda at_x_array: derivative_f(at_x_array, Ratio_Kde, CUDA) #Gradient of our function at cur_x
        
        while  iters < max_iters:
            prev_x_array = cur_x_array #Store current x value in prev_x
            # compute gradient vector at prev_x
            gradient_array = df_array(prev_x_array+new_origin_array)  # grantident computed based on the origin coordinate
            #normalize tangent vector 
            tangent_vec_array, prev_x_array_normalized, cur_x_array = SimporCuda.tangent_compute(gradient_array,prev_x_array,radius_array,lr )
            iters = iters+1 #iteration count
        return cur_x_array + new_origin_array ## return x on original origin coordinate

## if using multiple threads CPU
import multiprocessing
from multiprocessing import Pool
def thread_generate_synthetic_each_example(args):
    """ Kernel Multithreads finding optima and
        Generate k synthetic data points from each sample
        return : k synthetic samples
    """
    verbose = 0 # gradient ascent verbose
    debug  = 0 # printout x and maxima
    
    if debug: print("----Thread: {}".format(multiprocessing.current_process()))
    # extract arguments 
    X, y, x, label , minor_cls, major_cls, k, R, bandwidth, gd_max_iters,gd_lr, gd_tolerance,GD_patience,r_dist, kde_ratio, CUDA = args
     
    if label == minor_cls:  
        random_radius = np.random.uniform(low=0, high=R, size=1) 
        optima = gd(X, y, minor_cls= minor_cls, major_cls= major_cls, h=bandwidth,radius=random_radius, x=x, \
                            max_iters=gd_max_iters, lr=gd_lr, tolerance=gd_tolerance,GD_patience=GD_patience, kde_ratio=kde_ratio, verbose=verbose, CUDA=CUDA)
        X_result=[] 
        y_result=[]        
        for j in range(k): #each maxima, generate k neighbors to speedup
            if 'beta' in r_dist:
                x_eps = np.array( optima + np.random.beta(a= float(r_dist.split('_')[-2]), b= float(r_dist.split('_')[-1]) , size= optima.shape )*R ) #BEta distribution for R
            elif 'gaussian' in r_dist:
                x_eps = np.array( optima + np.random.normal(loc=0.0, scale= R*float(r_dist.split('_')[-1]), size= optima.shape  )  )              
            else:
                x_eps = np.array( optima + np.random.uniform(low=0.0, high=R, size= optima.shape ) ) # add some small noise to enrich the data
            
            if debug: print("optima,  x, x_eps",optima, x, x_eps  )
            X_result.append(np.array(x_eps) )  
            y_result.append(np.array(minor_cls) )

        return np.array(X_result),np.array(y_result)
    else: return # return None -> need to filter out at the end

# Implementing using CUDA    
def generate_synthetic_samples_cuda(args):
    """
    Compute synthetic samples for all minority samples at once (CUDA version)
    x is now a multidimentional array 
    """
    verbose = 0 # gradient ascent verbose
    debug  = 0 # printout x and maxima


    # extract arguments 
    X, y, x, label , minor_cls, major_cls, k, R, bandwidth, gd_max_iters,gd_lr, gd_tolerance,GD_patience,r_dist, kde_ratio, CUDA = args

    # prepare and get each synthetic sample for every samples 
    if (label == minor_cls).all():  
        random_radius = np.array([np.random.uniform(low=0, high=r, size=1) for r in R]) 
        #find a optima synthetic data for every minority samples
        optima_array = gd(X, y, minor_cls= minor_cls, major_cls= major_cls, h=bandwidth,radius=random_radius, x=x, \
                            max_iters=gd_max_iters, lr=gd_lr, tolerance=gd_tolerance,GD_patience=GD_patience, kde_ratio=kde_ratio, verbose=verbose, CUDA=CUDA)
        X_result=[] 
        y_result=[]        
        
        for optima,r in zip(optima_array,R):
            for j in range(k): #each maxima, generate k neighbors to speedup
                x_eps = np.array( optima + np.random.uniform(low=0.0, high=r/10, size= optima.shape ) ) # add some small noise to enrich the data
                if debug: print("optima,  x, x_eps",optima, x, x_eps  )
                X_result.append(np.array(x_eps) )  
                y_result.append(np.array(minor_cls) )

        return np.array(X_result),np.array(y_result)
    else: return None # return None -> need to filter out at the end



def balance_subset(big_X, big_y, sub_X, sub_y, major_cls, minor_cls_list,k , bandwidth ,kde_ratio,  n_threads, CUDA, gd_args):
    """
    Balancing a sub set of data sub_X, sub_y 
    big_X, big_y: entire dataset and labels X,y 
    sub_X, sub_y: the sub considering dataset needed to be balanced 
    k: to speedup, each optima can be used to generate k synthetic samples
    bandwidth: bandwidth of gaussian kernel for KDE
    kde_ratio: boolean , whether to maximize ratio f0/f1 or only f0
    n_threads: number of CPU threads
    CUDA: it is not supported yet
    gd_args: params for gradient asenct to find maximum posterior ratio
    Return: Synthetic samples 
    """
    k_R_distance = gd_args[0] # number of neighbers to compute max radius R
    iter_max = gd_args[1] #max iteration for Gradient Ascent to find optima
    lr = gd_args[2] #lerning rate for gradient ascent method
    tolerance = gd_args[3] #consider as a increasement in step size 
    GD_patience  = gd_args[4]
    r_dist = gd_args[5]
    print("Sub set of data label count:",Counter(sub_y),"major cls:",major_cls )
    print("Finding optima for Simpor:Gradient ascent iter_max:",iter_max,", lr:",lr )
    # Prepare the new balanced dataset D' 
    X_temp = big_X[:1] # Take only first samples as a initial array 
    y_temp = big_y[:1]
    
    #go through minority labels in the subset
    for c in minor_cls_list: # if the sample belongs to minority classes
        # count the labels in subset
        major_cls_count = len(sub_y[sub_y==major_cls] ) 
        minor_cls_count = len(sub_y[sub_y==c] )  
        #sub set in sub_X belonging to minor class
        
        sub_X_minor =  sub_X[sub_y==c]
        sub_y_minor =  sub_y[sub_y==c] 
        # count number of examples need to be generated
        num_generate = max(major_cls_count - minor_cls_count,0)
        #sample from the minority subset 
        print("Generate for class :{}  Num of generate {}".format(c,num_generate) )

        #go through exemples in the subset with label c
        if num_generate != 0 and len(sub_X_minor)!= 0: 
            num_generated = 0
            # generate random indices for sampling in subset 
#             length= len(sub_X_minor)
#             idx_list = list(range(length))*(num_generate//length) + list(range(length))[0:num_generate%length]
#             rand_example_indices = idx_list
            rand_example_indices =  np.random.choice(len(sub_X_minor), int(num_generate/k) ) 
            
            #compute max radius R based on average of K-nearest neighbors distances
            R_all_points=[]
            neighbor = NearestNeighbors(n_neighbors=k_R_distance, radius=3, n_jobs=n_threads)
            neighbor.fit(big_X)
            for idx in rand_example_indices:
                nbrs = np.array(neighbor.kneighbors([sub_X_minor[idx]], min(k_R_distance,len(big_X))\
                                           ,  return_distance=True) ) 
                
                # reject condition: if number of minor neighbors  < 1/3 number of major points
                nb_dis, nb_idx = nbrs
                # print("*****  nb_idx : {}".format(nb_idx))
                # print("*****  range(k_R_distance) : {}".format(range(k_R_distance)))
                nb_labels = [ big_y[int(nb_idx[0,i])] for i in range(k_R_distance)  ]
                nb_labels_dict = Counter(nb_labels)
                if nb_labels_dict[c] < 1/5 * nb_labels_dict[major_cls] : break 

                nbrs_dist = np.array([d for d in nbrs[:,0][0,1:-1] ]) # exclude the x itself
                R_all_points.append(nbrs_dist.mean() )
            R_all_points = np.array(R_all_points)
            print("------number of threads: {}".format( n_threads))
            if not CUDA:
                # if R points reject too many, then each point create k samples to make up
                if len(R_all_points) != 0: k = round(num_generate / len(R_all_points)) +1 
                args = [(big_X, big_y, sub_X_minor[idx] , sub_y_minor[idx], c ,major_cls , k, R, \
                            bandwidth, iter_max , lr, tolerance,GD_patience,r_dist, kde_ratio, CUDA) \
                        for idx,R in zip(rand_example_indices,R_all_points)] 
                
                with Pool(n_threads ) as p:
                    r = p.map(thread_generate_synthetic_each_example, args)

                r = list(filter(None,r)) ## remove None values
                for r_x,r_y in r:
                    if num_generated >= num_generate: break
                    X_temp = np.concatenate(  (X_temp,r_x), axis=0) 
                    y_temp = np.concatenate(  (y_temp,r_y), axis=0) 
                    num_generated += len(r_y) # count number of generated samples so far 
                # sanity check
                # if num_generated < num_generate: breakpoint() 

                return X_temp[1:major_cls_count+1], y_temp[1:major_cls_count+1]
            else:
                args = (big_X, big_y, sub_X_minor[rand_example_indices] , sub_y_minor[rand_example_indices],\
                        c ,major_cls , k, R_all_points, \
                        bandwidth, iter_max , lr, tolerance,GD_patience,r_dist, kde_ratio, CUDA) 
                results = generate_synthetic_samples_cuda(args)
                if results is not None:
                    return results[0], results[1] 
                else:return X_temp[0:1], y_temp[0:1]
        else:return X_temp[0:1], y_temp[0:1]
                    
        
def max_FracPosterior_balancing(X, y, k=1 , h = 0.1, AL_classifier=None, informative_threshold = 0.4,  n_threads=1, CUDA= False, gd_args=None):
    """
    Balancing data based on maximizing Posterior ratio
    
    Params
    -------
    X,y : entire data set
    k: number of synthetic samples for each time finding an optima(duplicate samples to speed up)
    h: gaussian kernel bandwidth
    AL_classifier: a trained classifier with the raw data for sampling high entropy samples 
    informative_threshold: Entropy based uncertainty threashold 
    n_threads: number of CPU threads
    CUDA: it is not supported yet 
    
    Return
    ----------
    X_2prime, y_2prime : fully balanced dataset D'
    X_prime, y_prime : informative balanced dataset (for investigation purpose)
    """
    
    label_dict = Counter(y)
    print("Start max_FracPosterior_balancing...")
    print("Label Dict entire data: ",label_dict)
    print("label keys ", list(label_dict.keys()) )
    print("Local balancing for  X,y shapes", X.shape, y.shape)
    
    ## M is the majority class (scalar)
    ## V is a list of minority classes (list)
    M = max(label_dict, key=label_dict.get)
    V = [key for key in label_dict.keys() if key not in [M]]
    print("major class ",M,"\nMinority class(es) :", V)


    threshold = informative_threshold       
    selected_indices = uncertainty_sampling(X, threshold , AL_classifier)  
    X_iftive = X[selected_indices]
    y_iftive = y[selected_indices]
    #the remaining region
    remain_idx = [indx for indx in range(len(X)) if indx not in selected_indices]
    X_remain = X[remain_idx]
    y_remain = y[remain_idx]

    ## if using cuda, initiate cuda
    if CUDA:
        print("***RUNING ON CUDA DEVICE****")
        from cuda_simpor import  SimporCuda
        import cupy as cp
        global  SC 
        SC = SimporCuda(X,y,h)
        

    ##start balancing informative set
    X_iftive_synthetic, y_iftive_synthetic = balance_subset(X, y, X_iftive, y_iftive, M, V, k , h, True, n_threads, CUDA, gd_args)
    #concatenate original data with informative synthetic data
    X_prime = np.concatenate( (X,X_iftive_synthetic) ,axis=0) ; y_prime = np.concatenate( (y,y_iftive_synthetic),axis=0) 
    label_prime_cnt  = Counter(y_prime)
    print("Finish informative balancing, X_prime shape, y_prime shape, labelcount", X_prime.shape, y_prime.shape, label_prime_cnt) 

    ###uncomment if want to balance the remaining data            
    ## globally balancing the remainder of data
    print("Start balancing the remainder of data")
    X_remain_synthetic, y_remain_synthetic = balance_subset(X, y, X_remain, y_remain, M, V, k , h, True, n_threads, CUDA,gd_args)
    # concatenate all data
    X_2prime = np.concatenate( ( X_prime ,X_remain_synthetic) ,axis=0) ; y_2prime = np.concatenate( (y_prime,y_remain_synthetic),axis=0) 
    label_2prime_cnt  = Counter(y_2prime)
    print("Finish global balancing, X_2prime shape, y_2prime shape, labelcount", X_2prime.shape, y_2prime.shape, label_2prime_cnt)
    return X_2prime, y_2prime ,X_prime, y_prime
    # return X_prime, y_prime ,X_prime, y_prime
   



