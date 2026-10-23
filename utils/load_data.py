from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_subject(path, num_classes=12):
    a=np.load(path,allow_pickle=False)
    if a.ndim!=2 or a.shape[1]<5: raise ValueError(f"{path}: expected [id,X,Y,Z,label]")
    x=np.asarray(a[:,1:4],dtype=np.float32)
    raw=a[:,4]
    if not len(raw) or not np.isfinite(x).all() or not np.isfinite(raw).all():
        raise ValueError(f"{path}: missing or nonfinite values")
    if not np.array_equal(raw,raw.astype(np.int64)): raise ValueError("Labels must be integers")
    y=raw.astype(np.int64)
    if y.min()<0 or y.max()>=num_classes: raise ValueError("Label out of range")
    return x,y

def generate_data(X,y,sequence_lenght=10,step=1):
    X=np.asarray(X,dtype=np.float32);y=np.asarray(y)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y): raise ValueError("Mismatched X/y")
    if sequence_lenght<1 or step<1: raise ValueError("Invalid window/step")
    limits=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    windows=[];labels=[]
    for start,end in zip(limits[:-1],limits[1:]):
        for i in range(int(start),int(end)-sequence_lenght+1,step):
            windows.append(X[i:i+sequence_lenght]);labels.append(int(y[start]))
    if not windows:
        return np.empty((0,sequence_lenght,X.shape[1]),np.float32),np.empty(0,np.int64)
    return np.stack(windows),np.asarray(labels,dtype=np.int64)

def windows_from_recording(x,y,sequence_length,overlap,num_classes):
    if not 0<=overlap<1: raise ValueError("overlap must be in [0,1)")
    stride=max(1,round(sequence_length*(1-overlap)))
    w,labels=generate_data(x,y,sequence_length,stride)
    return w,np.eye(num_classes,dtype=np.float32)[labels]
