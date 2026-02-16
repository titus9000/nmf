import numpy as np
from nmf import kl_linear_mu,kl_linear_grad,kl_div
import matplotlib.pyplot as plt

v = np.array([1,2,3])
W = np.array([[1,1],[2,1],[3,1]])
h = np.array([2,2])

fig,(ax1,ax2)=plt.subplots(1,2)
h , cost = kl_linear_mu(v,W,tol=1e-12,niter=1000)
ax1.plot(cost)
h,cost=kl_linear_grad(v,W,tol=1e-12,niter=100)
ax2.plot(cost)
plt.show()