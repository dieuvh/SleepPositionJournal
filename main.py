"""Static time-series classifier. --k_fold 0 = fixed person split; >=2 = person CV."""
import argparse,copy,csv,importlib,json,random
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from utils.load_data import read_subject,windows_from_recording

REGISTRY={
"lstm":("lstm","LSTM","lstm"),
"adaptive_lstm":("adaptive_lstm","AdaptiveLSTM","adaptive_lstm"),
"cnn":("cnn","CNN","cnn"),
"mlp":("mlp","MLP","mlp"),
"baseline":("lstm_baseline","LSTM_baseline","lstm_baseline"),
"transformers":("transformer","Transformer","transformers"),
"cnn_bilstm":("cnn_bilstm","CNNBiLSTM","cnn_bilstm"),
"cnn_bilstm_att":("cnn_bilstm_att","CNNBiLSTMAttention","cnn_bilstm"),
"cnn_bilstm_gru":("cnn_bilstm_gru","CNNBiLSTMGRU","cnn_bilstm_gru"),
"cnn_gru":("cnn_gru","CNNGRU","cnn_bilstm"),
"cnn_gru_att":("cnn_gru_att","CNNGRUAttention","cnn_gru_att"),
"cnn_gru_bilstm":("cnn_gru_bilstm","CNNGRUBiLSTM","cnn_gru_bilstm"),
"cnn_gru_bilstm_att":("cnn_gru_bilstm_att","CNNGRUBiLSTMAttention","cnn_gru_bilstm_att"),
}
def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model_type",choices=REGISTRY,default="adaptive_lstm")
    p.add_argument("--scenario",choices=["person_divide"],default="person_divide")
    p.add_argument("--data_type",choices=["static"],default="static")
    p.add_argument("--k_fold",type=int,default=0)
    p.add_argument("--num_classes",type=int,default=12)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--sequence_length",type=int,default=100)
    p.add_argument("--overlap",type=float,default=0.4)
    p.add_argument("--batch_size",type=int,default=512)
    p.add_argument("--normalizer",default="batch_norm")
    p.add_argument("--loss_fn",choices=["ce","nll"],default="ce")
    p.add_argument("--filters",type=int,default=8)
    p.add_argument("--kernel_size",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--learning_rate",type=float,default=0.001)
    p.add_argument("--validation_ratio",type=float,default=0.2)
    p.add_argument("--patience",type=int,default=20)
    p.add_argument("--train_dir",default="data/new_data_static/trainset")
    p.add_argument("--test_dir",default="data/new_data_static/testset")
    p.add_argument("--data_dir",default="data/new_data_static/all_person")
    p.add_argument("--output_dir",default="results")
    return p.parse_args()

def build(o):
    mod,cls,cfg=REGISTRY[o.model_type]
    klass=getattr(importlib.import_module("model."+mod),cls)
    config=copy.deepcopy(getattr(importlib.import_module("config."+cfg),"Config")())
    config.timestep=o.sequence_length;config.n_classes=o.num_classes
    config.normalizer=o.normalizer
    for name,value in (("filters",o.filters),("filter",o.filters),("kernel_size",o.kernel_size)):
        if hasattr(config,name):setattr(config,name,value)
    return klass(config).build()

def arrays(paths,scaler,o):
    xs=[];ys=[]
    for p in paths:
        x,y=read_subject(p,o.num_classes)
        xx,yy=windows_from_recording(scaler.transform(x),y,o.sequence_length,o.overlap,o.num_classes)
        if len(yy):xs.append(xx);ys.append(yy)
    if not xs:raise ValueError("No valid windows; reduce --sequence_length")
    return np.concatenate(xs),np.concatenate(ys)

def run(train,val,test,o,folder,seed):
    import tensorflow as tf
    random.seed(seed);np.random.seed(seed)
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    records=[read_subject(p,o.num_classes)[0] for p in train]
    scaler=MinMaxScaler().fit(np.concatenate(records,axis=0))
    xt,yt=arrays(train,scaler,o);xv,yv=arrays(val,scaler,o);xe,ye=arrays(test,scaler,o)
    folder.mkdir(parents=True,exist_ok=True)
    model=build(o)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=o.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])
    callbacks=[
      tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=o.patience,restore_best_weights=True),
      tf.keras.callbacks.ModelCheckpoint(str(folder/"best.keras"),monitor="val_loss",save_best_only=True)]
    h=model.fit(xt,yt,validation_data=(xv,yv),batch_size=o.batch_size,
                epochs=o.epochs,callbacks=callbacks,verbose=2)
    pred=np.argmax(model.predict(xe,batch_size=o.batch_size,verbose=0),axis=1)
    true=np.argmax(ye,axis=1)
    result={"accuracy":float(accuracy_score(true,pred)),
            "macro_f1":float(f1_score(true,pred,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(true,pred,average="weighted",zero_division=0))}
    (folder/"metrics.json").write_text(json.dumps(result,indent=2))
    (folder/"history.json").write_text(json.dumps(h.history,default=float,indent=2))
    (folder/"classification_report.txt").write_text(classification_report(true,pred,zero_division=0))
    np.save(folder/"confusion_matrix.npy",confusion_matrix(true,pred,labels=np.arange(o.num_classes)))
    print(result);return result

def main():
    o=args()
    if o.k_fold<0 or o.k_fold==1:raise ValueError("--k_fold must be 0 or >=2")
    if not 0<=o.overlap<1:raise ValueError("Invalid overlap")
    if not 0<o.validation_ratio<1:raise ValueError("Invalid validation_ratio")
    base=Path(o.output_dir)/o.model_type
    if o.k_fold:
        subjects=sorted(Path(o.data_dir).glob("*.npy"))
        if len(subjects)<o.k_fold:raise ValueError("Not enough subjects")
        rows=[]
        for fold,(tr,te) in enumerate(KFold(n_splits=o.k_fold,shuffle=True,random_state=o.seed).split(subjects),1):
            if len(tr)<2:raise ValueError("Need >=2 subjects for inner validation")
            ti,vi=train_test_split(tr,test_size=o.validation_ratio,random_state=o.seed+fold)
            files=lambda ix:[subjects[i] for i in ix]
            print(f"fold={fold} train={len(ti)} val={len(vi)} test={len(te)}")
            rows.append({"fold":fold,**run(files(ti),files(vi),files(te),o,base/f"fold_{fold}",o.seed+fold)})
        base.mkdir(parents=True,exist_ok=True)
        with (base/"fold_summary.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        summary={k:{"mean":float(np.mean([r[k] for r in rows])),"std":float(np.std([r[k] for r in rows]))}
                 for k in ("accuracy","macro_f1","weighted_f1")}
        (base/"kfold_summary.json").write_text(json.dumps(summary,indent=2))
        print(summary)
    else:
        tr=sorted(Path(o.train_dir).glob("*.npy"))
        te=sorted(Path(o.test_dir).glob("*.npy"))
        if len(tr)<2 or not te:raise ValueError("Need >=2 training subject files and >=1 test subject")
        ti,vi=train_test_split(np.arange(len(tr)),test_size=o.validation_ratio,random_state=o.seed)
        run([tr[i] for i in ti],[tr[i] for i in vi],te,o,base/"single",o.seed)
if __name__=="__main__":main()
