from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
from torch.nn.init import xavier_normal_
import numpy as np

from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks
        

class RESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1];
        self.rhs = self.embeddings[0]

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank);
        rhs = self.rhs(x[:, 2])

        return (torch.bmm(lhs.unsqueeze(1), rel)).squeeze() @ self.rhs.weight.t(), [(lhs, rel, rhs)]


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:3]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:3]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size


        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[2]


    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return (lhs * rel) @ self.rhs.weight.t(), [(lhs, rel, rhs)]
    



class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]);rrel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]; 
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), [
                   (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]



class RotatE(KBCModel):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ
       Direct implementation following the style of the ComplEx model in models.py."""

    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(RotatE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x: torch.Tensor):
        """
        Compute scores and regularization factors using rotations.

        Args:
            x (torch.Tensor): A tensor of shape [batch_size, 3] containing (head, relation, tail) indices.

        Returns:
            score (torch.Tensor): The computed scores.
            factors (list of tuple): Regularization factors for (head, relation, tail) embeddings.
        """
        # Get head, relation, and tail embeddings
        head = self.embeddings[0](x[:, 0])
        rel  = self.embeddings[1](x[:, 1])
        tail = self.embeddings[0](x[:, 2])

        # Split embeddings into real and imaginary parts
        head_real, head_imag = head[:, :self.rank], head[:, self.rank:]
        rel_real, rel_imag   = rel[:, :self.rank], rel[:, self.rank:]
        tail_real, tail_imag = tail[:, :self.rank], tail[:, self.rank:]

        # Normalize relation embeddings to obtain rotation parameters
        rel_norm = torch.sqrt(rel_real ** 2 + rel_imag ** 2)
        cos = rel_real / rel_norm
        sin = rel_imag / rel_norm

        # Rotate head embeddings using relation parameters
        rotated_head_real = head_real * cos - head_imag * sin
        rotated_head_imag = head_real * sin + head_imag * cos

        # Use all entity embeddings as targets; split them into real and imaginary parts
        all_entities = self.embeddings[0].weight
        all_entities_real, all_entities_imag = all_entities[:, :self.rank], all_entities[:, self.rank:]

        # Compute the RotatE score by taking dot products with all target embeddings
        score = (rotated_head_real @ all_entities_real.transpose(0, 1) +
                 rotated_head_imag @ all_entities_imag.transpose(0, 1))

        # Compute regularization factors (norms) for head, relation, and tail embeddings
        # factors = [(torch.sqrt(head_real ** 2 + head_imag ** 2),
        #             torch.sqrt(rel_real ** 2 + rel_imag ** 2),
        #             torch.sqrt(tail_real ** 2 + tail_imag ** 2))]
        factors = [(head, rel, tail)]
        return score, factors



class TuckER(KBCModel):
    def __init__(self, sizes: tuple, rank: int, init_size: float = 1e-3):
        """
        sizes: (num_entities, num_relations, num_entities)
        rank: embedding dimension for both entities and relations
        init_size: scaling factor for initialization
        """
        super(TuckER, self).__init__()
        num_entities, num_relations, _ = sizes

        self.rank = rank
        self.E = nn.Embedding(num_entities, rank)
        self.R = nn.Embedding(num_relations, rank)
        # Core tensor W of shape (rank, rank, rank)
        self.W = nn.Parameter(torch.Tensor(rank, rank, rank))
        # Initialize W uniformly between -1 and 1
        nn.init.uniform_(self.W, a=-1, b=1)

        self.loss = BCEWithLogitsLoss()

        # Scale embeddings as in other models
        self.E.weight.data *= init_size
        self.R.weight.data *= init_size

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (batch_size, 3) with [head, relation, tail] indices.
        Returns:
            scores: Tensor of shape (batch_size, num_entities) with predicted scores.
            extras: A list containing a tuple (head embeddings, relation embeddings, tail embeddings).
        """
        # Lookup embeddings for head, relation, and tail
        head = self.E(x[:, 0])  # (B, rank)
        rel = self.R(x[:, 1])   # (B, rank)
        tail = self.E(x[:, 2])  # (B, rank)
        
        # Compute relation-specific transformation matrices.
        # For each sample i, we compute:
        #   W_mat[i] = sum_j rel[i,j] * W[j,:,:]
        # Resulting in a tensor of shape (B, rank, rank)
        W_mat = torch.einsum('bi,ijk->bjk', rel, self.W)
        
        # Transform head embeddings using the relation-specific matrices.
        # head.unsqueeze(1): (B, 1, rank), bmm with (B, rank, rank) -> (B, 1, rank)
        head_transformed = torch.bmm(head.unsqueeze(1), W_mat).squeeze(1)  # (B, rank)
        
        # Compute scores for all entities by dotting with all entity embeddings.
        # self.E.weight: (num_entities, rank) -> transpose gives (rank, num_entities)
        scores = torch.matmul(head_transformed, self.E.weight.t())
        
        # For consistency with other models, return tail embeddings as extras.
        return torch.sigmoid(scores), [(head, rel, tail)]


