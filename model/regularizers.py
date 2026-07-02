from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torch.linalg import svdvals


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass
class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(f ** 2)/ f.shape[0]            
        return norm 
class Fro(Regularizer):
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.norm(f, 2) ** 2
                )
        return norm / factors[0][0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm


class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        # print("factorts:",factors.shape)
        for factor in factors:
            # print("factor:",factor.shape)
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 2
                )
        return norm / factors[0][0].shape[0]

class L1(Regularizer):
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f)**1
                )
        return norm / factors[0][0].shape[0]

class NA(Regularizer):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()



class ER(Regularizer):
    def __init__(self, weight: float):
        super(ER, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a=1;b=1#can be adjusted
        rate=0.455 # can be adopted from{0.5-1.5}  0.5
        scale=0.6# can be adopted from{0.5-1.5}  0.75

        for factor in factors:
            h, r, t = factor
            norm += rate*torch.sum(t**2 + h**2)
            #t=torch.abs(t);h=torch.abs(h);r=torch.abs(r)
            if a==b:
                norm+=scale*torch.sum(a*h**2*r**2+a*t**2*r**2+b*h**2*r**2+b*t**2*r**2)
            else:
                norm += scale*torch.sum(a*h**2 *r**2 +a*2*h *r*t*r+ a*t**2 *r**2+b*h**2 * r**2 -b*2*h * r*t*r+ b*t**2 * r**2)
        return self.weight * norm / h.shape[0]
    

    
class DURA(Regularizer):
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        for factor in factors:
            h, r, t = factor

            norm += torch.sum(t**2 + h**2)
            norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]


class DURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 2 + t ** 2)
            norm += torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]
class Spiral3(DURA):
    def __init__(self, weight: float, lambda_1=1.0, lambda_2= 1.0, lambda_3= 1.1):
        super(Spiral3, self).__init__(weight)
        self.lambda_1 = lambda_1  # Weight for higher-order interactions
        self.lambda_2 = lambda_2  # Weight for consistency regularization
        self.lambda_3 = lambda_3  # Weight for sparsity regularization

        def forward(self, factors):
            norm = 0
            total_norm = 0

            # Calculate the mean norm of all embeddings
            for factor in factors:
                h, r, t = factor
                total_norm += torch.norm(h) + torch.norm(r) + torch.norm(t)

            mean_norm = total_norm / (3 * len(factors))

            for factor in factors:
                h, r, t = factor

                # Adaptive weight based on embedding norms
                alpha = (torch.norm(h) + torch.norm(r) + torch.norm(t)) / mean_norm

                # Original DURA regularization
                norm += torch.sum(t**2 + h**2)
                norm += torch.sum(h**2 * r**2 + t**2 * r**2)

                # Adaptive weight on regularization
                norm += alpha * (torch.sum(t**2 + h**2) + torch.sum(h**2 * r**2 + t**2 * r**2))

            return self.weight * norm / h.shape[0]






class GeoReg(Regularizer):
    def __init__(self, weight: float, lambda_manifold: float = 0.1, tau: float = 0.5, lambda_align: float = 0.5):
        
        super(GeoReg, self).__init__()
        self.weight = weight
        self.lambda_manifold = lambda_manifold
        self.tau = tau
        self.lambda_align = lambda_align

    def forward(self, factors):
        reg = 0.0
        # Process each factor (triplet) independently.
        for factor in factors:
            h, r, t = factor  # h, r, t: [batch_size, d]

            dura_term = (torch.sum(h**2) + torch.sum(t**2) +
                         torch.sum((h * r)**2) + torch.sum((t * r)**2))
            
            # ----- Alignment term -----
            align_term = torch.sum((h - t)**2)
            
            # ----- Manifold regularization (Laplacian term) -----
            # Here we apply it separately for heads and tails.
            batch_size = h.shape[0]
            
            h_norm = h / (torch.norm(h, dim=1, keepdim=True) + 1e-8)
            cos_sim_h = torch.mm(h_norm, h_norm.t())  # [batch_size, batch_size]
            W_h = torch.exp(cos_sim_h / self.tau)
            diff_h = h.unsqueeze(1) - h.unsqueeze(0)  # [batch_size, batch_size, d]
            diff_h_sq = torch.sum(diff_h ** 2, dim=2)   # [batch_size, batch_size]
            laplacian_h = torch.sum(W_h * diff_h_sq) / (batch_size ** 2)
            
            t_norm = t / (torch.norm(t, dim=1, keepdim=True) + 1e-8)
            cos_sim_t = torch.mm(t_norm, t_norm.t())
            W_t = torch.exp(cos_sim_t / self.tau)
            diff_t = t.unsqueeze(1) - t.unsqueeze(0)
            diff_t_sq = torch.sum(diff_t ** 2, dim=2)
            laplacian_t = torch.sum(W_t * diff_t_sq) / (batch_size ** 2)
            
            manifold_term = self.lambda_manifold * (laplacian_h + laplacian_t)
            
            reg += dura_term + self.lambda_align * align_term + manifold_term
        
        batch_size = factors[0][0].shape[0]
        return self.weight * reg / batch_size


class ER_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(ER_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a=1;b=1# can be adjusted. e.g., a=1 b=1.02
        rate=1 # can be adopted from{0-1.5}  1  1.05
        scale=0.5# can be adopted from{0-1.5}  0.5  0.455
        for factor in factors:
            h, r, t = factor
            norm += rate*torch.sum(h ** 2 + t** 2)  
            if a==b:
                norm += scale*torch.sum(
                a*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))**2+a*torch.bmm(r, t.unsqueeze(-1)) **2+ b*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))**2+b*torch.bmm(r, t.unsqueeze(-1)) **2)                
            else:
                norm += scale*torch.sum(
                a*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))**2+2*a*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))*torch.bmm(r, t.unsqueeze(-1))+a*torch.bmm(r, t.unsqueeze(-1)) **2+ b*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))**2-2*b*torch.bmm(r.transpose(1, 2),h.unsqueeze(-1))*torch.bmm(r, t.unsqueeze(-1))+b*torch.bmm(r, t.unsqueeze(-1)) **2)
        return self.weight * norm / h.shape[0]
    
# ----------------------------------
class SPR(Regularizer):
    def __init__(self, weight: float, delta=0.9):
        super(SPR, self).__init__()
        self.weight = weight
        self.delta = delta

    def select_small(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        
        # Flatten x and sort the values in ascending order.
        x_flat = x.view(-1)
        sorted_vals, indices = torch.sort(x_flat)
        cumulative = torch.cumsum(sorted_vals, dim=0)

        S = (cumulative <= delta).sum()
        mask_flat = torch.zeros_like(x_flat)
        if S > 0:
            mask_flat[indices[:S]] = 1.0
        mask = mask_flat.view_as(x)
        return mask

    def forward(self, factors: tuple) -> torch.Tensor:
       
        norm = 0.0
        for factor in factors:
            h, r, t = factor

            h2 = h ** 2
            t2 = t ** 2
            hr2 = h2 * (r ** 2)
            tr2 = t2 * (r ** 2)

            mask_h = self.select_small(h2, self.delta)
            mask_t = self.select_small(t2, self.delta)
            mask_hr = self.select_small(hr2, self.delta)
            mask_tr = self.select_small(tr2, self.delta)

            h2_sparse = h2 * (1 - mask_h)
            t2_sparse = t2 * (1 - mask_t)
            hr2_sparse = hr2 * (1 - mask_hr)
            tr2_sparse = tr2 * (1 - mask_tr)

            norm += torch.sum(h2_sparse + t2_sparse)
            norm += torch.sum(hr2_sparse + tr2_sparse)

        batch_size = factors[0][0].shape[0]
        return self.weight * norm / batch_size
    
class ASPR(Regularizer):  
    def __init__(self, weight: float, tau0=1e-1, epsilon: float = 1e-2, gamma: float = 1e-5):
        
        super(SPR, self).__init__()
        self.weight = weight
        self.tau0 = tau0
        self.epsilon = epsilon
        self.gamma = gamma

    def soft_weight(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute the soft threshold weight for each element in x:
            w(x) = σ((x - tau)/epsilon)
        """
        return torch.sigmoid((x - tau) / self.epsilon)
    
    def forward(self, factors: tuple) -> torch.Tensor:
        
        total_loss = 0.0
        for factor in factors:
            h, r, t = factor  # assume shape: (batch_size, embed_dim)
            
            # Compute squared terms.
            h2 = h ** 2
            t2 = t ** 2
            hr2 = h2 * (r ** 2)
            tr2 = t2 * (r ** 2)
            
           
            signal = 0.5 * (h2.mean() + t2.mean())
            tau = self.tau0 * (signal ** self.gamma)
            
            w_h = self.soft_weight(h2, tau)
            w_t = self.soft_weight(t2, tau)
            w_hr = self.soft_weight(hr2, tau)
            w_tr = self.soft_weight(tr2, tau)
            
            h2_weighted = h2 * w_h
            t2_weighted = t2 * w_t
            hr2_weighted = hr2 * w_hr
            tr2_weighted = tr2 * w_tr
            
            triplet_loss = torch.sum(h2_weighted, dim=1) \
                           + torch.sum(t2_weighted, dim=1) \
                           + torch.sum(hr2_weighted, dim=1) \
                           + torch.sum(tr2_weighted, dim=1)
            total_loss += torch.sum(triplet_loss)

            
        
        # Average over the batch.
        batch_size = factors[0][0].shape[0]
        return self.weight * total_loss / batch_size


class AdpSPR(Regularizer):
   
    def __init__(self,
                 model: nn.Module,        # <-- reference to the model
                 weight= 1e-1,
                 tau0= 1e-2,
                 epsilon: float = 1e-2,
                 gamma: float = 0.5):
        """
        
        """
        super(AdpSPR, self).__init__()
        self.model = model
        self.weight = weight
        self.tau0 = tau0
        self.epsilon = epsilon
        self.gamma = gamma

    def _compute_grad_norm(self) -> float:
       
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                g = param.grad.data
                total_norm += g.norm(2).item() ** 2
        return total_norm ** 0.5

    def _soft_weight(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        
        return torch.sigmoid((x - tau) / self.epsilon)

    def forward(self, factors: tuple) -> torch.Tensor:
        
        grad_norm = self._compute_grad_norm()

        tau = self.tau0 * (grad_norm ** self.gamma)

        total_loss = 0.0
        for (h, r, t) in factors:
            h2 = h ** 2
            t2 = t ** 2
            hr2 = h2 * (r ** 2)
            tr2 = t2 * (r ** 2)

            w_h  = self._soft_weight(h2,  tau)
            w_t  = self._soft_weight(t2,  tau)
            w_hr = self._soft_weight(hr2, tau)
            w_tr = self._soft_weight(tr2, tau)

            h2_weighted  = h2 * w_h
            t2_weighted  = t2 * w_t
            hr2_weighted = hr2 * w_hr
            tr2_weighted = tr2 * w_tr

            triplet_loss = torch.sum(h2_weighted + t2_weighted + hr2_weighted + tr2_weighted, dim=1)
            total_loss  += torch.sum(triplet_loss)

        batch_size = factors[0][0].shape[0]
        return self.weight * total_loss / batch_size

