import numpy as np
from nmd import kl_nmd_mu,kl_nmd_grad,kl_div

W=np.array([[1,2],[.02,.01],[1,1]])
h=np.array([1.5,.5])

v=W.dot(h)
# v=random.exponential(v)

# h,cost=kl_nmd_mu(v,W,tol=1e-16,niter=1000)
# h,cost=kl_nmd_grad(v,W,tol=1e-5,niter=1)

h=np.ones(W.shape[1])
Wh=W.transpose()
vhat=W.dot(h)
d= Wh.dot(v/vhat)-Wh.sum(axis=1)
print([kl_div(v,W.dot(h+s*d)) for s in np.arange(.01,.02,.005)])
print("Stop here : there is a problem with nmd_mu : the cost was going up instead of down. Problem of sign ? no, probably constants to adjust (rho,rho_desc, rho_asc) ça fait un sacre boulot pour demontrer que cet algo se plante est ce qu il ne vaudrait pas mieux le faire theoriquement ? mais c est toujours plus satisfaisant de l'avoir sous la main")
#print(cost)
#fig,ax=plt.subplots(1,figsize=(15,5))
#plt.plot(cost)
#plt.yscale('log')
#plt.grid('on')
#plt.show()