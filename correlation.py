import numpy as np 
def corr(x,y): 
    cov = ((x-x.mean())*(y-y.mean())).mean()
    cc_abs = cov/np.sqrt(x.var()*y.var())
    return cc_abs