from numpy.linalg import norm
from numpy import log, ones, array,ones_like,maximum,tile,shape,logical_and,logical_or
from scipy import random
from scipy import stats

def is_div(y,x):
    e=y/x
    return (e-log(e)-1).mean()

def is_nmd_mu(v,W,h=None,niter=100,tol=1e-4):
    it=0
    cost=[]
    
    if h is None:
        h=ones(W.shape[1])
    Wh=W.transpose()
        
    while True:
        vhat=W.dot(h)
        cost.append(is_div(v,vhat))

        if it>=niter or cost[it]<tol:
            break

        h = h*(Wh.dot(v/(vhat**2)))/(Wh.dot(1/vhat))
        it+=1
        
    return h,array(cost)


def kl_div(y,x):
    return (y*log(y/x)+x-y).mean()

def kl_linear_mu(v,W,h=None,niter=100,tol=1e-4):
    '''This is the version with h and v as vectors'''
    it=0
    cost=[]
    
    if h is None:
        h=ones(W.shape[1])
    Wh=W.transpose()
        
    while True:
        vhat=W.dot(h)
        cost.append(kl_div(v,vhat))

        if it>=niter or cost[it]<tol:
            break

        h = h*(Wh.dot(v/vhat))/Wh.sum(axis=1)
        it+=1
        
    return h,array(cost)

def kl_linear_mod(v,W,h=None,eps=1e-4,niter=100,tol=1e-4):
    it=0
    cost=[]

    if h is None:
        h=ones(W.shape[1])
    Wh=W.transpose()

    while True:
        vhat=W.dot(h)
        cost.append(kl_div(v,vhat))

        if it>=niter or cost[it]<tol:
            break
    
        h=maximum((h+eps)*(Wh.dot(v/vhat))/Wh.sum(axis=1)-eps)
        it += 1
    
    return h,array(cost)

def kl_nmd_mu(V,W,H=None,niter=100,tol=0,eps=0):
    it=0
    cost=[]
    
    if eps>0:
        V = V + eps

    if H is None:
        H=ones((W.shape[1],V.shape[1]))

    Wh=W.transpose()

    Ones = ones_like(V)

    while True:

        Vhat = eps + W.dot(H)

        cost.append(kl_div(V,Vhat))

        if it>=niter or cost[it]<tol:
            break

        H = H*(Wh.dot(V/Vhat))/Wh.dot(Ones)

        it+=1
        
    return H,array(cost)

def kl_nmd_mod(V,W,H,**kwargs):
    if "adjust" in kwargs and kwargs["adjust"]:
        return kl_nmd_mod_adjust(V,W,**kwargs)
    else:
        return kl_nmd_mod_raw(V,W,**kwargs)

def kl_nmd_mod_raw(V,W,H=None,niter=100,tol=0,eps=0,eta=0):
    it = 0
    cost = []
    K = W.shape[1]
    N=V.shape[1]
    
    if H is None :
        H = tile(V.mean(0)/K,[K,1])
    
        
    if eps>0:
        V = V + eps

    Wt = W.transpose()
    Vhat = eps + W.dot(H)
    
    cost.append(kl_div(V,Vhat))

    while True:


        Ones = ones_like(V)
        
        H = maximum((H+eta)*(Wt.dot(V/Vhat))/Wt.dot(Ones)-eta,0)

        Vhat = eps + W.dot(H)
        
        cost.append(kl_div(V,Vhat))

        it+=1

        if it >= niter or cost[it] < tol:
            break
    
    return H,array(cost)

def kl_nmd_sparse(V,W,H,niter=100,sparsity=0,tol=0,eps=0,eta=0):
    it = 0
    cost = []
    K = W.shape[1]
    N=V.shape[1]
    
    if H is None :
        H = tile(V.mean(0)/K,[K,1])
    
        
    if eps>0:
        V = V + eps

    Wt = W.transpose()
    Ones = ones_like(V)
    Q = Wt.dot(Ones)+sparsity
    Vhat = eps + W.dot(H)
    
    cost.append(kl_div(V,Vhat))

    while True:
        
        H = maximum((H+eta)*(Wt.dot(V/Vhat))/Q-eta,0)

        Vhat = eps + W.dot(H)
        
        cost.append(kl_div(V,Vhat))

        it+=1

        if it >= niter or cost[it] < tol:
            break
    
    return H,array(cost)

def kl_nmd_mod_adjust(V,W,H=None,niter=100,tol=0,eps=0,eta=0,adjust=True,decay=None):
    it = 0
    cost = []
    K = W.shape[1]
    N=V.shape[1]
    
    if H is None :
        H = tile(V.mean(0)/K,[K,1])
    
    if eps>0:
        V = V + eps

    if decay is None:
        decay = .9
        
    Wt = W.transpose()
    Vhat = eps + W.dot(H)

    cost.append(kl_div(V,Vhat))
    if adjust and len(shape(eta))==0:
        eta = tile(1e-4,[K,N])

    while True:

        Ones = ones_like(V)
        ntry = 0

        while True:
            eta = eta*decay

            G = (Wt.dot(V/Vhat))/Wt.dot(Ones)
            active = logical_or(H>0,G>1)
            H1 = H.copy()
            H1[active] = (H1[active]+eta[active])*G[active]-eta[active]
            H1[H1<0]=0

            Vhat1 = eps+W.dot(H1)
            oc = cost[-1]
            c = kl_div(V,Vhat1)
            if c < oc :
                break
        
        Vhat= Vhat1
        H = H1
        cost.append(c)

        it+=1

        if it >= niter or cost[it] < tol :
            break
    
    return H,array(cost)
    
def nmd_init(V,W,method='center'):
    # For future use : 
    # initialize with result of linear solve
    if method=='ones':
        H = ones((W.shape[1],V.shape[1]))
    elif method=='mean scaled':
        H = tile(V.mean(0)/K,[K,1])
    elif method=='center':
        H=tile(1/K,[K,1])
    return H

def kl_linear_grad(v,W,h=None,niter=100,tol=1e-4):
    it=0
    cost=[]
    
    # will be replaced by nmd_init
    if h is None:
        h=ones(W.shape[1])

    Wh = W.transpose()
    
    def cp_cost(h):
        vh=W.dot(h)
        return kl_div(v,vh)

    #parameters of the line search 

    def linesearch(h):
        s=.01
        rho=.99
        rho_desc=.9
        rho_asc=10
        # outline : 
        # compute descent direction
        # then perform line search à la Armijo
        vhat=W.dot(h)
        # compute the descent direction
        d= Wh.dot(v/vhat)-Wh.sum(axis=1)
        c=cp_cost(h)
        breakoff = False
        has_iter=False
        while cp_cost(h+s*d)>c*rho:
            s=s*rho_desc
            if norm(s*d)<1e-10:
                breakoff=True
                break
        
        if not(breakoff) and not(has_iter):
            while cp_cost(h+s*d)<=c*rho:
                s=s*rho_asc
        return h+s*d, breakoff
    
    breakoff=False
    cost.append(cp_cost(h))

    while True:        
        h,breakoff = linesearch(h)
        it+=1
        cost.append(cp_cost(h))
        # arrêt si variation trop faible
        if it>=niter or (cost[it-1]-cost[it]<tol) or breakoff:
            break
        
    return h,array(cost)
