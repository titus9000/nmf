import numpy as np
import matplotlib.pyplot as plt
from scipy import random
from scipy import stats
from math import log,sqrt,log10
from nmf import kl_nmd_mu,kl_nmd_mod

W=np.array([[1,1],[2,1],[3,1]])
H=np.array([[1,1,1],[0,1,2]])

V=np.array([[1,2,3],[2,3,4],[3,4,5]])

H0 = np.array([[2/3,2,8/3],[1/3,1,4/3]])
H,cost  = kl_nmd_mu( V,W,H=H0,niter=1000)
H2,cost2= kl_nmd_mod(V,W,H=H0,niter=1000,eta=.1)

# fig, axes=plt.subplots(1,2,figsize=(12,5))
# axes[0].plot(cost)
# axes[0].set_yscale('log')
# axes[0].set_xscale('log')

# axes[1].plot(cost)
# axes[1].set_yscale('log')
# axes[1].set_xscale('log')

fig,ax=plt.subplots(1)
ax.plot(cost,label='mu')
ax.plot(cost2,label='mu mod')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()