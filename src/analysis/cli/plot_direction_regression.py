import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import xyz
from analysis.geometry import opening_angle
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--predictions',default='outputs/preds.npy');p.add_argument('--targets',default='outputs/targets.npy');p.add_argument('--indices',default='outputs/indices.npy');p.add_argument('--output-dir',type=Path,default=Path('plots/direction'));p.add_argument('--output-file',type=Path,default=None);p.add_argument('--dpi',type=int,default=200);p.add_argument('--inverse-component-norm',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args();o=load_outputs(a.run_dir,a.predictions,a.targets,a.indices);pred=xyz(o.predictions);true=xyz(o.targets);pred=-1+2*pred if a.inverse_component_norm else pred;true=-1+2*true if a.inverse_component_norm else true;error=opening_angle(true,pred);xlabel='Opening angle [deg]';name='opening_angle.png';q=np.quantile(error,.68);print(f'Events: {len(error):,}; 68% quantile = {q:.4f}');fig,ax=plt.subplots(figsize=(7,5));ax.hist(error,bins=100,histtype='step');ax.axvline(q,c='crimson',ls='--',label=f'68% = {q:.3f}');ax.set(xlabel=xlabel,ylabel='Events');ax.grid(alpha=.3);ax.legend();save(fig,output_path(a.output_dir,a.output_file or name),a.dpi)
if __name__=='__main__':main()
