from pathlib import Path
import matplotlib.pyplot as plt

def save(fig,path,dpi=200):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig);print(f'Saved: {path}')
