import numpy as np

def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(n==0):raise ValueError('Zero-length direction vector')
    return v/n
def opening_angle(true,pred):return np.degrees(np.arccos(np.clip(np.sum(unit(true)*unit(pred),axis=-1),-1,1)))
def wall(v,radius=1690.,half_height=1810.):
    v=np.asarray(v); return np.minimum(radius-np.hypot(v[:,0],v[:,1]),half_height-np.abs(v[:,2]))
def towall(v,d,radius=1690.,half_height=1810.):
    v=np.asarray(v,float);d=unit(d);x,y,z=v.T;dx,dy,dz=d.T;a=dx*dx+dy*dy;b=2*(x*dx+y*dy);c=x*x+y*y-radius*radius;disc=np.clip(b*b-4*a*c,0,None);cand=np.full((len(v),4),np.inf);m=a>0;cand[m,0]=(-b[m]+np.sqrt(disc[m]))/(2*a[m]);cand[m,1]=(-b[m]-np.sqrt(disc[m]))/(2*a[m]);m=np.abs(dz)>0;cand[m,2]=(half_height-z[m])/dz[m];cand[m,3]=(-half_height-z[m])/dz[m];cand[cand<=0]=np.inf;return cand.min(1)
