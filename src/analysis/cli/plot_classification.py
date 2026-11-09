import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save

def labels(x):
 x=np.asarray(x);return x.astype(int) if x.ndim==1 else (x[:,0].astype(int) if x.shape[1]==1 else np.argmax(x,1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--predictions',default='outputs/preds.npy');p.add_argument('--targets',default='outputs/targets.npy');p.add_argument('--indices',default='outputs/indices.npy');p.add_argument('--output-dir',type=Path,default=Path('plots/classification'));p.add_argument('--output-file',type=Path,default=Path('classification_confusion_matrix.png'));p.add_argument('--dpi',type=int,default=200);a=p.parse_args();o=load_outputs(a.run_dir,a.predictions,a.targets,a.indices);y=labels(o.targets);pred=np.argmax(o.predictions,1);n=o.predictions.shape[1];names=['muon','electron','pion'][:n];cm=confusion_matrix(y,pred,labels=np.arange(n));norm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,float),where=cm.sum(1,keepdims=True)!=0);print(f'Accuracy: {np.mean(y==pred):.6f}');fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(n),yticks=range(n),xticklabels=names,yticklabels=names,xlabel='Predicted class',ylabel='True class');
 for i in range(n):
  for j in range(n):ax.text(j,i,f'{norm[i,j]:.3f}',ha='center',va='center')
 fig.colorbar(im,ax=ax);fig.tight_layout();save(fig,output_path(a.output_dir,a.output_file),a.dpi)
if __name__=='__main__':main()
