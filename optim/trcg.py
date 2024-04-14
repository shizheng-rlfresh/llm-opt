"""
Preconditioned Trust Region Newton-CG (Conjugate Gradient) Method

"""


from collections.abc import Mapping
from typing import Optional, Union, List
from transformers import PreTrainedModel
import torch
from torch import nn
import numpy as np
import time

class TRCG:
    """
    TRCG is a simple implementation of Preconditioned Newton-CG Method. This is a state-of-the-art optimizer for non-convex problem.
    Yet, under non-deterministic setting, i.e., stochastic setting, such as using mini-batch to train a machine learning model, the convergence
    is not fully backed by theory. 


    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`):
              The model to fine-tune, loaded either in quantized version or not, with LoRA model enabled
        device ([`torch.device`]):
              default to torch.device("cpu")
        radius (`int`, *optional*):
              trust region radius
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        device: Optional[torch.device] = torch.device("cpu"),
        radius: Optional[int] = 0.1,
        eval_BS: Optional[int] = 32,
        cgopttol: Optional[float] = 1e-7,
        cgmaxiter: Optional[float] = 20,
        c0tr: Optional[float] = 0.2,
        c1tr: Optional[float] = 0.25,
        c2tr: Optional[float] = 0.75,
        t1tr: Optional[float] = 0.25,
        t2tr: Optional[float] = 2.0,
        radius_max: Optional[float] = 5.0,
        radius_initial: Optional[float] = 0.1,
    ):
        
        self.model = model
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = cgmaxiter 
        self.iterationCounterForAdamTypePreconditioning = 0
        self.DiagPrecond = [w.data*0.0 for w in self.model.parameters() if w.requires_grad]
        self.DiagScale = 0.0
        self.eval_BS = eval_BS
        self.grad_bulk = [] 
        self.BS = 0 
        self.newLOSS = 0.0
        
    def findroot(self,x,p) -> float:
        
        aa = 0.0
        bb = 0.0
        cc = 0.0
    
        for e in range(len(x)):
            aa += (p[e]*p[e]).sum()
            bb += (p[e]*x[e]).sum()
            cc += (x[e]*x[e]).sum()
        
        bb = bb*2.0
        cc = cc - self.radius**2
    
        alpha = (-2.0*cc)/(bb+(bb**2-(4.0*aa*cc)).sqrt())

        return alpha.item()
    
    def computeListNorm(self, lst: List[torch.tensor]) -> float:
        return np.sum([(li.data*li.data).sum().item() for li in lst])**0.5
    
    def computeListNormSq(self, lst: List[torch.tensor]) -> float:
        return  np.sum([ (li.data*li.data).sum().item() for li in lst]) 
    
    def CGSolver(self, loss_grad: List[torch.tensor], cgloop: List[float] = []):
        
        # preconditioner
        scl = np.sqrt((1 - np.power(self.DiagScale, self.iterationCounterForAdamTypePreconditioning)))
        self.SquaredPreconditioner = [1.0/torch.sqrt(di)*scl  for di in self.DiagPrecond]
        
        #
        # use previously computed loss_grad for initial setup
        #
        cg_iter = 0 # iteration counter
        x0 = [i.data*0 for i in self.model.parameters() if i.requires_grad]
        # r0 = [i.data+0.0 for i in loss_grad]  # set initial residual to gradient
        # p0 = [-i.data+0.0 for i in loss_grad] # set initial conjugate direction to -r0
        # self.cgopttol = self.computeListNormSq(loss_grad)
        # self.cgopttol = self.cgopttol**0.5
        # self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
        
        r0 = [(i.data+0.0)*pr.data for i, pr in zip(loss_grad, self.SquaredPreconditioner)]
        p0 = [-(i.data+0.0)*pr.data for i, pr in zip(loss_grad, self.SquaredPreconditioner)]
        self.cgopttol = self.computeListNormSq(r0)
        self.cgopttol = self.cgopttol**0.5
        self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
        
        cg_term = 0
        j = 0
        
        #
        # CG iterations
        #
        
        while 1:
            j+=1
            self.CG_STEPS_TOOK = j
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j=j-1
                p1 = x0
                break
                
            #
            # compute gradient and Hessian-vector product iteratively for each CG iteration
            #
            cgloopstart = time.time()
            
            param_requires_grad = [w for w in self.model.parameters() if w.requires_grad]
            
            # hessian vector product
#             loss_grad_direct = torch.sum(torch.stack([(gi*si).sum() for gi, si in zip(loss_grad,p0)]))
#             HP = torch.autograd.grad(loss_grad_direct, param_requires_grad, retain_graph=True)
            
            loss_grad_direct \
            = torch.sum(torch.stack([(gi*(si*pr.data)).sum() for gi, si, pr in zip(loss_grad, p0, self.SquaredPreconditioner)]))
            HP = torch.autograd.grad(loss_grad_direct, param_requires_grad, retain_graph=True)
            HP = [ g*pr.data for g, pr in zip(HP, self.SquaredPreconditioner)]
            
            PHP = np.sum([(Hpi*p0i).sum().item() for Hpi, p0i in zip(HP,p0)])
            
            cgloop.append(time.time() - cgloopstart)
            
            # if nonpositive curvature detected, go for the boundary of trust region
            if PHP <= 0:
                tau = self.findroot(x0,p0)
                p1 = [xi+tau*p0i  for xi, p0i in zip(x0,p0)]
                cg_term = 1
                break
            
            # if positive curvature
            # vector product
            rr0 = self.computeListNormSq(r0)

            # update alpha
            alpha = rr0/PHP
            
            x1 = [xi+alpha*pi for xi,pi in zip(x0,p0)]
            norm_x1 = self.computeListNorm(x1)
            
            # if norm of the updated x1 > radius
            if norm_x1 >= self.radius:
                tau = self.findroot(x0,p0)
                p1 = [xi+tau*pi for xi,pi in zip(x0,p0)]
                cg_term = 2
                break
    
            # update residual
            r1 = [ri+alpha*HPi for ri, HPi in zip(r0, HP)]
            norm_r1 = self.computeListNorm(r1)
    
            if norm_r1 < self.cgopttol:
                p1 = x1
                cg_term = 3
                break
    
            rr1 = self.computeListNormSq(r1)
            beta = rr1/rr0
    
            # update conjugate direction for next iterate
            p1 = [-ri+beta*pi for ri,pi in zip(r1,p0)]
    
            p0 = p1
            x0 = x1
            r0 = r1
    

        cg_iter = j
        norm_p1 = self.computeListNorm(p1)
        p1 = [pi*pr.data for pi,pr in zip(p1, self.SquaredPreconditioner)]
        d = p1

        return d, cg_iter, cg_term, cgloop, norm_p1
    
    def step(self, batch: Mapping, oldloss: int, loss_grad: List[torch.tensor], sample=None):
        f_cost=0 # fun eval
        g_cost=0 # grad and/or Hv 
        
        for gi in loss_grad:
            if gi.grad_fn is None:
                raise ValueError('no grad_fn found in %s'%repr(gi))
        
        CGITER = 0.0

        w0 = [w.data+0.0 for w in self.model.parameters() if w.requires_grad]
        
        firstloopstart = time.time()
        
        firstloop = [time.time() - firstloopstart]
       
        
            
        self.DiagScale = 0.99
        self.iterationCounterForAdamTypePreconditioning += 1

        for gi, di in zip(loss_grad, self.DiagPrecond):
            di.data.copy_(di.data*self.DiagScale+(1-self.DiagScale)*torch.abs(gi.data))
            di.data[di.data==0]+=1.0
        
        #
        # update preconditioner
        # 
        secondloop = [] 
        thirdloop = []
        cgloop=[]

        # Conjugate Gradient Method
        d, cg_iter, cg_term, cgloop, norm_p1 = self.CGSolver(loss_grad, cgloop)

        CGITER += cg_iter
        g_cost+=cg_iter

        #
        # compute gradient and hessian-vector product for determining ratio
        #
        secondloopstart = time.time()
        
        param_requires_grad = [w for w in self.model.parameters() if w.requires_grad]
        
        loss_grad_direct = torch.sum(torch.stack([(gi*di).sum() for gi, di in zip(loss_grad, d)]))

        Hd = torch.autograd.grad(loss_grad_direct, param_requires_grad, retain_graph=True)
        DHD = np.sum([(Hdi*di).sum().item() for Hdi, di in zip(Hd, d)])
        GD = np.sum([(gi.data*di).sum().item() for gi,di in zip(loss_grad, d)])
        g_cost+=1.0

        secondloop.append(time.time() - secondloopstart)

        # update model parms
        with torch.no_grad():
            for wi,di in zip(param_requires_grad, d):
                wi.add_(di+0.0)

        thirdloopstart = time.time()
        
        with torch.no_grad():
            self.newLOSS = self.model(**batch).loss.item()
            
        f_cost+=1.0
        thirdloop.append(time.time()-thirdloopstart)
        norm_d = self.computeListNorm(d)

        numerator = oldloss - self.newLOSS

        denominator = -GD - 0.5*DHD

        # ratio
        rho = numerator/denominator
        update = 3 # default reject
        if rho < self.c1tr: # shrink radius
            self.radius = self.t1tr*self.radius
            update = 0
        if rho > self.c2tr and np.abs(norm_p1 - self.radius) < 1e-10: # enlarge radius
            self.radius = min(self.t2tr*self.radius,self.radius_max)
            update = 1
        # otherwise, radius remains the same
        if rho <= self.c0tr or np.isnan(rho): # reject d
            update = 3
            with torch.no_grad():
                for wi,w0i in zip(param_requires_grad, w0):
                    wi.set_(w0i+0.0)
                    
        return d, rho, update, CGITER, cg_term, loss_grad, norm_d, norm_p1, numerator, denominator,\
               firstloop, secondloop, thirdloop, cgloop, f_cost, g_cost