

#  need tensorflow environment
import builtins as __builtin__
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles,make_classification
from sklearn.utils.multiclass import unique_labels
import sklearn
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
import pandas as pd

import pickle
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from simpor import max_FracPosterior_balancing

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import datetime
import math
from DATA.creditcardfraud.load_data import load_creditcard
from load_KEEL_UCI import load_KEEL, load_UCI
from collections import Counter
from plotResult import plotResult

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


log_dir_name = 'LOG_10'
test_size_percent = 0.1 ## developement only
exp_methods = ['SIMPOR']

CONTINUE = True


def log_prepare(dataset_name):
    global log_dir, log_file, figure_path, result_dir, checkpoint, expdir
    expdir = dataset_name+str(datetime.datetime.now()).replace(":",'.').replace(" ","-")

    if CONTINUE: 
        with open('tested_exp.txt') as file:
            for line in file:
                if dataset_name in line:
                    expdir = line.rstrip()
    log_dir = "./{}/".format(log_dir_name) + expdir
    log_file = os.path.join(log_dir,"log.txt")
    figure_path = os.path.join(log_dir,"Figures")
    result_dir = os.path.join(log_dir,"Results")
    checkpoint = os.path.join(log_dir,"Checkpoint")
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if not os.path.isdir(figure_path): os.makedirs(figure_path)
    if not os.path.isdir(result_dir): os.makedirs(result_dir)
    if not os.path.isdir(checkpoint): os.makedirs(checkpoint)

def print(*args, **kwargs):
    """My custom print() function.
       Print out and also keep loging 
    """
    with open(log_file, "a") as log:  
        log.write("\n"+str(datetime.datetime.now())+": ")
        log.writelines(args)
    return __builtin__.print(*args, **kwargs)

def data_gen(dataset_name, IR, args):
    """
    Generate synthetic imbalanced dataset
    dataset_name: 'breast_cancer', 'moon', 'creditcard', 'mnist'
    IR: artificial Imabalance Ratio 
    -----
    Return 
    X_train, X_test, y_train, y_test
    """
    global k, k_neighbors, n_neuron,epoch , n_layers, lr
    global X, labels, gd_args

    ##
    UCI = ['spectrometer', 'isolet' ,'us_crime', 'scene' , 'libras_move', 'arrhythmia' ,'solar_flare_m0' ,'oil','car_eval_4','wine_quality','ozone_level','yeast_me2']
    ## default DNN params
    lr =0.01

    ##Gradient ascent params to find maximum posterior ratio 
    k_R_distance = 10# number of neighbers to compute max radius R
    iter_max = 500 #max iteration for Gradient Ascent to find optima
    GD_lr = 0.00001 #lerning rate for gradient ascent method
    tolerance = 0.0001 #consider as a increasement in step size 
    GD_patience = 10 # gradient ascent patience for stoping early
    gd_args = [k_R_distance,iter_max,GD_lr,tolerance,GD_patience, args.r_dist   ]
    

    if dataset_name in UCI:
        X,labels =load_UCI(dataset_name)
    else:# KEEL dataset
        X,labels  = load_KEEL(dataset_name)
    n_neuron = 100
    n_layers = 3
    lr = 0.1
    epoch = 200
    cls_need_removal = [0]  # majority
    rm_ratio = [ 0 ] #removal percentage
    k =  1  # for maxFracPosterior, generate k neighbors for each found minima in informative set
    k_neighbors = 5 #for SMOTE

    print("\r==================={}===================\n\n".format(dataset_name))
          .format(RandomSeed, n_neuron, n_layers, epoch,k,cls_need_removal))
    print("Gradient Ascent Parameters:\n K_R_distance : {}\n Iter_max : {}\n GD_lr : {}\n Tolerance: {}\n GD_patience: {}".format(k_R_distance,iter_max,GD_lr,tolerance, GD_patience))
    ##scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = (X- np.mean(X,axis=0) )/ np.std(X,axis=0)
    ##shuffle data
    from sklearn.utils import shuffle
    X, labels = shuffle(X, labels)

    plotboundary(None, X , labels , model_name="dnn", title='', save='OriginalRawData', border=False)
    
    
    c=Counter(labels)
    print("Total original data {}".format(c))


    ###remove some sample for unbalance 
    for cls,ratio in zip(cls_need_removal,rm_ratio):
        cnt = int(len(labels[labels==cls])*ratio) #remove rm_ratio of train data
        rm=[]
        for i,l in enumerate(labels):
            if l==cls and cnt>0 :
                rm.append(i)
                cnt -=1
        X = np.delete(X, rm, axis=0)
        labels = np.delete(labels, rm, axis=None)


    c_total_after_rm = Counter(labels);
    print("Total counter after Imabalnce Removal: {}".format(c_total_after_rm))
    c_0 = c_total_after_rm[0.0]
    c_1 = c_total_after_rm[1.0]

   
    
    ####SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state = RandomSeed)
    plotboundary(None, X_train, y_train , model_name="dnn", title='', save='ImbalancedData', border=False)
    print("X_train: {}".format( X_train.shape))
    print("y_train: {}".format( y_train.shape) )
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format( y_test.shape))
    print("Counting the number of samples in each class")
    c_train = Counter(y_train); print("Train set {}".format(c_train))
    c_test = Counter(y_test);   print("Test set {}".format(c_test ))
    # plt.show()
    print("*** Imabalance Ratio (IR): {}\n".format( c_0/c_1 if c_0 > c_1 else c_1/c_0 )  )

    ## remove nan column
    X_train = X_train[:, ~np.isnan(X_train).any(axis=0)]
    X_test = X_test[:,~np.isnan(X_test).any(axis=0)]

    # ##checking data validation
    if np.any(np.isnan(X_train)) : 
        nan_indx = np.argwhere(np.isnan(X_train))
        print("*****{}".format(X_train[nan_indx[0]]))
        raise ValueError('There is Nan or Inf in data at indices:{}'.format(nan_indx))
    return X_train, X_test, y_train, y_test

##create imbalanced dataset by removing samples in minority classes
## testing on widely-used methods


def normalize1d(X):   
    return (X - X.min() ) / (( X.max() - X.min() ) + 0.00000001)

def plotboundary(clf, X,y, model_name="DNN", title='',save='',border=True):
    input_dim= X.shape[1]             
    # create a mesh to plot in
    x0 = X[:, 0]
    x1 = X[:, 1]
    x_min, x_max =  x0.min() - 0.2, x0.max() + 0.2
    y_min, y_max =  x1.min() - 0.2, x1.max() + 0.2
    if input_dim==2 and border:
        h = .02  # step size in the mesh   
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
    # Put the result into a color plot  
    fig = plt.figure()
    plt.scatter(x0, x1, s=7, c = y, cmap = 'winter')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    if input_dim==2 and border: 
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired, linewidths=2.5)

    plt.axis('off')
    plt.title(title)
    plt.savefig(figure_path + "/"+save+".jpg")
#     plt.show()
    

## create simple deep fully connected neural network
class My_classifier():   
    def __init__(self,epochs = 200, n_neuron= 300, batch_size=64,n_layers=3):      
        # Create a MirroredStrategy.## model checkpoint
        strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=2, verbose=2, epsilon=1e-4,min_lr=0.001,  mode='auto')
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)
            ##early stop callback
            early_stop =  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,restore_best_weights=True)
            Callbacks = [ reduce_lr_loss, model_checkpoint_callback, early_stop]
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            self.epochs = epochs
            self.batch_size=batch_size
            self.callbacks= Callbacks
            self.model = Sequential()
            self.model.add(Dense(n_neuron, input_dim=input_dim, activation='relu'))
            for i in range(n_layers-1):
                # self.model.add(Dropout(0.2))
                # self.model.add(Dense(n_neuron,kernel_regularizer='l1_l2',  activation='relu'))
                self.model.add(Dense(n_neuron, activation='relu'))
            self.model.add(Dense(2, activation='softmax'))
            # compile the keras model
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
            self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            
    def fit(self,X,y, X_val, y_val, sample_weight = None):     
        y = to_categorical(y, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        if sample_weight is not None:
            return self.model.fit(X, y, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks= self.callbacks, sample_weight= sample_weight )
        else:
            return self.model.fit(X, y,validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks= self.callbacks)
    def predict_proba(self,X):
        return self.model.predict(X)
    def predict(self,X):
        return np.argmax(self.model.predict(X),axis=-1)

    
def loss_plot(history, file_path, title):
    history_loss_path = file_path.replace('.png','') + '.csv'
    # print("history.history.keys(): {}".format( history.history.keys()) ) 
    # keys history : ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    data = [history.history[k] for k in history.history.keys()]
    df = pd.DataFrame(data)
    df.to_csv(history_loss_path, index=False)

def data_save(X,y, file_path):
    #save data 
    data_path = file_path.replace('png','csv').replace("train loss","data")
    c = Counter(y)
    print("- Saving data to csv files.")
    print("- X train shape, y train shape : {} {}".format(X.shape, y.shape ))
    print("- Balanced Train data Counter :{}".format(c))
    y_expanded = np.expand_dims(y,axis=1)
    df = pd.DataFrame(np.concatenate( (X,y_expanded ),axis=1  ) )
    df.to_csv(data_path, index=False)

def n_classification(X_train, y_train,X_val,y_val, X_test, y_test, n=1, epoch=120,**kwargs):
    """
    Classify n times and  return the average
    n : number of runs
    epoch: training epochs
    return 
    F1Scores: array of n f1 scores 
    """ 
    method = kwargs.get("method")
    num_runs = kwargs.get("num_runs")
    model = kwargs.get("classifier")
    if method is not None and num_runs is not None:
        no_log = False
    else: no_log = True
    
    F1Scores = []
    pred_array = []
    for i in range(n):   
        if method=="ee": # if using classifier of method (forexample ensemble classifier)
            classifier = model
            history =  classifier.fit(X_train, y_train)
        else:
            classifier = My_classifier(epochs=epoch,n_neuron=n_neuron,n_layers=n_layers)
            history = classifier.fit(X_train, y_train, X_val, y_val)
        y_pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)[:,1]
        pred_array.append(y_pred_prob)
        report = classification_report(y_test, y_pred, digits=4, output_dict= True)
        F1Scores.append(report['macro avg']['f1-score'] )
        #plot
        title = "{}-train loss-run-{}-{}".format(method,num_runs,i)
        loss_path = os.path.join(result_dir,title+'.png')
        #save loss plot
        loss_plot(history,loss_path, title )
        data_save(X_train, y_train, loss_path)
    if method is not None and num_runs is not None:
        plotboundary(classifier, X_train ,y_train , model_name="dnn", title='', save='Training_Data_PLot_{}_run_{}.{}_F1_{:.4f}'.format(method,num_runs,i,F1Scores[-1]))
        plotboundary(classifier, X_val ,y_val , model_name="dnn", title='', save='Testing_Data_PLot_{}_run_{}.{}_F1_{}'.format(method,num_runs,i, F1Scores[-1]))

    ##logging
    if not no_log:
        #adding groundtruth at the last column
        pred_array.append(y_test)
        predict_log_path = os.path.join(result_dir,"{}-pred_and_label.{}-{}.csv".format(method,num_runs,i))
        #save prediction
        df = pd.DataFrame(pred_array)
        df.to_csv(predict_log_path,index=False)

    return np.array(F1Scores),classifier

def Method_n_runs(**kwargs):
    """
    Run different times, 
    """
    method = kwargs.get("method")
    X_train = kwargs.get("X_train"); y_train = kwargs.get("y_train"); 
    X_test = kwargs.get("X_test"); y_test = kwargs.get("y_test");
    num_runs = kwargs.get("num_runs")
    epoch = kwargs.get("epoch")
    F1 = []
    SIMPOR_F1_informative = []
    SIMPOR_F1_full = []
    model = None
    randomSplit = True # random split data each test
    for i in range(num_runs): 
        rand = RandomSeed+i+1
        if randomSplit: X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state = rand)
        t = time.time()

        if method == "SIMPOR":
            k = kwargs.get("k"); h = kwargs.get("h"); p = kwargs.get("p"); AL_classifier = kwargs.get("AL_classifier"); 
            n_threads = kwargs.get("n_threads"); CUDA = kwargs.get("CUDA")
            X_train_new, y_train_new, X_informative, y_informative = \
            max_FracPosterior_balancing(X_train, y_train,k=k, h=h, AL_classifier = AL_classifier, \
                                            informative_threshold=p, n_threads= n_threads, CUDA=CUDA, gd_args= gd_args)
            _,temp_X,_, temp_y = train_test_split(X_test, y_test, test_size=  test_size_percent, random_state = rand)
            X_train_new = np.concatenate((X_train_new, temp_X))
            y_train_new = np.concatenate((y_train_new, temp_y))
             
        temp_f1, _ = n_classification(X_train_new, y_train_new, X_test, y_test, X_test, y_test, 1, epoch, method=method,num_runs=i, classifier=model)
        F1.append(temp_f1.mean())
    return np.array(F1)



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

import argparse
if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="DataSets: moon, breast_cancer, creditcard, glass1, etc. ",
                        type=str, default="moon")
    parser.add_argument("--r_dist", help="Factor r distritbution, e.g., gaussian_1, gaussian_0.2, gaussian_0.6, etc. ",
                        type=str, default="gaussian_1")
    parser.add_argument("--n_threads", help="number of threads, default 34, If using CUDA -> n_threads is set to 1",
                        type=int, default=10)
    parser.add_argument("--n_runs", help="number of trials, default 5",
                        type=int, default=5)
    parser.add_argument("--randseed", help="Random seed to split data, default 99",
                        type=int, default=99)
    parser.add_argument("--entropyPortion", help="take highest entropy portion of data for informative balancing",
                        type=float, default=None)
    parser.add_argument("--IR", help="Imbalance Ratio, default =3",
                        type=int, default=3)
    parser.add_argument("--cuda", help="using cuda, boolean{True,False}, default True",
                        type=boolean_string, default=False)
    parser.add_argument("--onlySimpor", help="Run only simpor method",
                        type=boolean_string, default=False)
    parser.add_argument("--note", 
                        type=str, default='')
    parser.add_argument("--gridSearch", help="Find the best params for Simpor, boolean{True,False}, would take a while, default True",
                        type=boolean_string, default=False)
    args = parser.parse_args()
    
    global dataset_name, input_dim, X_train, X_test, y_train, y_test, RandomSeed
    dataset_name = args.dataset
    n_threads = args.n_threads
    num_runs = args.n_runs 
    RandomSeed = args.randseed
    IR = args.IR
    CUDA = args.cuda
    

    # data preparation
    log_prepare(dataset_name)
    X_train, X_test, y_train, y_test = data_gen(dataset_name,IR,args)
    input_dim = X_train.shape[1]
    
     
    ### Preliminary result
    org_classifier = SVC(gamma=2, C=1, probability=True)
    org_classifier.fit(X_train, y_train)
    y_pred = org_classifier.predict(X_test)
    acc =  accuracy_score(y_test, y_pred)
    print("\n\n\nReport On raw data \
        (This is one time running, check the stable result at the final test): ")
    print("Acc: {}".format(acc) )
    print("F1_score".format( str(f1_score(y_test, y_pred, average = 'macro')) ) )
    print(str(confusion_matrix( y_test, y_pred) ) )
    print(str(classification_report(y_test, y_pred, digits=4)) )  


    ###############################################
    #Searching for the best parameters  
    if args.entropyPortion:
         max_f1 = {'f1':0,'h':0.1,'p':args.entropyPortion}
    elif args.gridSearch:
        Plot= True
        h_list = [  0.1] # bandwidth of Gaussian kernel
        entropy_threshold_list = [0.2, 0.4, 0.6]
        runs  = len(h_list) * len(entropy_threshold_list)
        max_f1 = {'f1':0,'h':0,'p':0}
        print("\n\n------------------Finding best Params SIMPOR in {} runs-----------------------\n".format(runs))
        for h in h_list:
            for p in entropy_threshold_list:
                t= time.time()
                AL_classifier = org_classifier
                X_full, y_full, X_informative, y_informative = \
                max_FracPosterior_balancing(X_train, y_train,k=k, h=h, AL_classifier = AL_classifier, \
                                            informative_threshold=p, n_threads = n_threads, CUDA= CUDA, gd_args = gd_args)
                print("X_full shape: {}".format( str(X_full.shape) ) )    
                SIMPOR_F1_informative, classifier_informative = n_classification(X_informative, y_informative,X_test, y_test, X_test, y_test, n=1,  epoch=epoch)
                SIMPOR_F1_full, classifier_full =  n_classification(X_full, y_full, X_test, y_test,X_test, y_test, n=1, epoch= epoch)
                if SIMPOR_F1_full.mean() > max_f1['f1']: #update paras to find the best
                    max_f1['f1'] = SIMPOR_F1_full.mean(); max_f1['h']=h;  max_f1['p']=p; print("SIMPOR k={} Threshold={}  ".format(k, p))

    else:
        #default 
        max_f1 = {'f1':0,'h':0.1,'p':0.2} 

                        
    #######################################################


    print("\n\n====================Final Test on {} runs====================".format(str((num_runs)) ) )
    print("*NOTE: {}".format(args.note))
    h = max_f1['h']
    p = max_f1['p']
    AL_classifier = org_classifier
    F1_scores = {}
    proc_time = {}

    for exp_method in exp_methods:
        start = time.time()
        if exp_method == 'SIMPOR':
            print("SIMPOR k={} h = {} Threshold={}  ".format(k, h, p))
            SIMPOR_F1_full =  Method_n_runs(method = 'SIMPOR', X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, k=k, h=h, AL_classifier = AL_classifier, \
                                                    p=p,  num_runs=num_runs, epoch=epoch, n_threads=n_threads,  CUDA= CUDA)
            
            F1_scores['SIMPOR'] = SIMPOR_F1_full
            if args.onlySimpor: #if run only simpor
                break
        else:
            
            F1_score =  Method_n_runs(method = exp_method, X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)
            
            # update result
            F1_scores[exp_method] = F1_score
        
        # update time
        stop = time.time()
        proc_time[exp_method] = stop-start
    

### --------------------summary 
    print("\n\n=====Summary after {} runs====".format(num_runs))
    for exp_method in exp_methods:
        if exp_method in exp_methods:
            print("{}_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format(exp_method, F1_scores[exp_method].mean() , F1_scores[exp_method].std()  ,proc_time[exp_method]/60 ) )
    plotResult(log_dir_name,expdir, not args.onlySimpor)
