import torch
from torch import Tensor
from supar.modules.sparsemax import Sparsemax
from supar.modules.dropout import SharedDropout, IndependentDropout
from supar.structs import semiring, StructuredDistribution, DependencyCRF
# from lpsmap import TorchFactorGraph, DepTree, Budget


class CrossEntropy:
    struct_constructor = None
    torch_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def local(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        loss = -p.softmax(dim) * q.log_softmax(dim)
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))

    @staticmethod
    def local_hard(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        tgt = p.argmax(dim)
        tgt[~mask] = -100
        loss = CrossEntropy.torch_ce_loss(q.movedim(dim, 1), tgt)
        return loss.sum(list(range(1, loss.ndim)))

    @staticmethod
    def local_sparseall(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        sparsemax = Sparsemax(dim)
        loss = -sparsemax(p) * sparsemax(q).log()
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))
    
    @staticmethod
    def local_sparsetarget(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        sparsemax = Sparsemax(dim)
        loss = -sparsemax(p) * q.log_softmax(-1)
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))

    # @staticmethod
    # def local_structsparsetarget(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
    #     p_ = p.cpu()
    #     p = p_[:, 1:, 1:].clone()
    #     p.diagonal(dim1=1, dim2=2).copy_(p_[:, 1:, 0])
    #     p = p.share_memory_()

    #     fg = TorchFactorGraph()
    #     u = fg.variable_from(p)
    #     fg.add(DepTree(u, packed=True, projective=True))
    #     fg.solve()
    #     out = u.value

    #     dist = DependencyCRF()

    @staticmethod
    def exact(p: StructuredDistribution, q: StructuredDistribution):
        return p.cross_entropy(q)
    
    @staticmethod
    def exact_hard(p: StructuredDistribution, q: StructuredDistribution):
        tgt = p.argmax
        return -q.log_prob(tgt)

    @staticmethod
    def topk(p: StructuredDistribution, q: StructuredDistribution, k:int, no_rest: bool):
        topn_edges = p.topk(k)
        if not no_rest:  # in SeqVAT
            log_prob_p = p.log_prob(topn_edges, batched=True)
            log_prob_q = q.log_prob(topn_edges, batched=True)
            sumed_prob_p = log_prob_p.exp().sum(-1)
            sumed_prob_q = log_prob_q.exp().sum(-1)
            numeric_mask = (sumed_prob_q > 1) | (sumed_prob_p > 1)
            loss = (-log_prob_p.exp() * log_prob_q).sum(-1).masked_fill(numeric_mask, 0)
            remainder_sum_p = (1 - sumed_prob_p)
            remainder_sum_q = (1 - sumed_prob_q).clamp(1e-4)  # due to numerical precisions
            loss += (-remainder_sum_p * remainder_sum_q.log()).masked_fill(numeric_mask, 0)
        else: 
            prob_p = p.scores(topn_edges).softmax()
            log_prob_q = q.scores(topn_edges).logsoftmax()
            loss = -(prob_p * log_prob_q).sum()
        if any(loss < 0):
            raise ValueError('loss <0 detected')
        return loss
    
    @staticmethod
    def topk_fix(p: StructuredDistribution, q: StructuredDistribution, k:int, no_rest: bool):
        if p.scores.shape[1] <= 3 and k > 3: return torch.zeros(0, requires_grad=True, device=p.scores.device)
        if p.scores.shape[1] <= 4 and k > 7: return torch.zeros(0, requires_grad=True, device=p.scores.device)
        topn_edges = p.topk(k)
        if not no_rest:  # in SeqVAT
            log_prob_p = p.log_prob(topn_edges, batched=True)
            log_prob_q = q.log_prob(topn_edges, batched=True)
            sumed_prob_p = log_prob_p.exp().sum(-1)
            sumed_prob_q = log_prob_q.exp().sum(-1)
            numeric_mask = (sumed_prob_q > 1) | (sumed_prob_p > 1)
            loss = (-log_prob_p.exp() * log_prob_q).sum(-1).masked_fill(numeric_mask, 0)
            remainder_sum_p = (1 - sumed_prob_p)
            remainder_sum_q = (1 - sumed_prob_q).clamp(1e-4)  # due to numerical precisions
            loss += (-remainder_sum_p * remainder_sum_q.log()).masked_fill(numeric_mask, 0)
        else: 
            prob_p = p.scores(topn_edges).softmax()
            log_prob_q = q.scores(topn_edges).logsoftmax()
            loss = -(prob_p * log_prob_q).sum()
        if any(loss < 0):
            raise ValueError('loss <0 detected')
        return loss

    @staticmethod
    def marginal(p: StructuredDistribution, q: StructuredDistribution, mask: Tensor):
        p = p.marginals
        q = q.marginals
        loss = -p * q.clip(1e-9).log()
        loss[~mask] = 0
        loss = loss.sum(list(range(1, loss.ndim)))
        return loss

    @staticmethod
    def exact_sparsemax(p: StructuredDistribution, q: StructuredDistribution, mask: Tensor):
        p = p.backward(p.forward(semiring.SparsemaxSemiring).sum())
        q = q.marginals
        loss = -p * q.clip(1e-9).log()
        loss[~mask] = 0
        loss = loss.sum(list(range(1, loss.ndim)))
        return loss

    @classmethod
    def dispatch(cls, mode: str, p, q, **kwargs):
        if mode == 'none':
            return 0.
        if mode == 'local':
            return cls.local(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_hard':
            return cls.local_hard(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparseall':
            return cls.local_sparseall(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparsetarget':
            return cls.local_sparsetarget(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if isinstance(p, Tensor):
            p = cls.struct_constructor(p)
            q = cls.struct_constructor(q)
        if mode == 'marginal':
            return cls.marginal(p, q, kwargs['mask'])
        if mode == 'exact':
            return cls.exact(p, q)
        if mode == 'exact_hard':
            return cls.exact_hard(p, q)
        if mode == 'exact_sparsemax':
            return cls.exact_sparsemax(p, q, kwargs['mask'])
        if mode.startswith('top'):
            return cls.topk(p, q, int(mode[3:]), False)
        if mode.startswith('ftop'):
            return cls.topk_fix(p, q, int(mode[4:]), False)

    @classmethod
    def register_struct_constructor(cls, func):
        cls.struct_constructor = func


class KL:
    struct_constructor = None

    @staticmethod
    def local(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        loss = p.softmax(dim) * (p.log_softmax(dim) - q.log_softmax(dim))
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))

    @staticmethod
    def local_sparseall(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        sparsemax = Sparsemax(dim)
        p = sparsemax(p)
        q = sparsemax(q)
        loss = p * (p.log() - q.log())
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))
    
    @staticmethod
    def local_sparsetarget(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
        sparsemax = Sparsemax(dim)
        p = sparsemax(p) 
        loss = p * (p.log() - q.log_softmax(-1))
        loss[~mask] = 0.0
        return loss.sum(list(range(1, loss.ndim)))

    # @staticmethod
    # def local_structsparsetarget(p: Tensor, q: Tensor, mask: Tensor, dim=-1):
    #     p_ = p.cpu()
    #     p = p_[:, 1:, 1:].clone()
    #     p.diagonal(dim1=1, dim2=2).copy_(p_[:, 1:, 0])
    #     p = p.share_memory_()

    #     fg = TorchFactorGraph()
    #     u = fg.variable_from(p)
    #     fg.add(DepTree(u, packed=True, projective=True))
    #     fg.solve()
    #     out = u.value

    #     dist = DependencyCRF()

    @staticmethod
    def exact(p: StructuredDistribution, q: StructuredDistribution):
        return p.kl(q)

    @staticmethod
    def topk(p: StructuredDistribution, q: StructuredDistribution, k:int, no_rest: bool):
        topn_edges = p.topk(k)
        if not no_rest:  # in SeqVAT
            log_prob_p = p.log_prob(topn_edges, batched=True)
            log_prob_q = q.log_prob(topn_edges, batched=True)
            sumed_prob_p = log_prob_p.exp().sum(-1)
            sumed_prob_q = log_prob_q.exp().sum(-1)
            numeric_mask = (sumed_prob_q >= 1) | (sumed_prob_p >= 1)
            loss = (log_prob_p.exp() * (log_prob_p-log_prob_q)).sum(-1).masked_fill(numeric_mask, 0)
            remainder_sum_p = (1 - sumed_prob_p)
            remainder_sum_q = (1 - sumed_prob_q).clamp(1e-4)  # due to numerical precisions
            loss += (remainder_sum_p * (remainder_sum_p.log() - remainder_sum_q.log())).masked_fill(numeric_mask, 0)
        else: 
            log_prob_p = p.scores(topn_edges).logsoftmax(-1)
            log_prob_q = q.scores(topn_edges).logsoftmax(-1)
            loss = (log_prob_p.exp() * (log_prob_q - log_prob_q)).sum()
        if any(loss < 0):
            raise ValueError('loss <0 detected')
        return loss

    @staticmethod
    def marginal(p: StructuredDistribution, q: StructuredDistribution, mask: Tensor):
        p = p.marginals
        q = q.marginals
        loss = p * (p.clip(1e-9).log() - q.clip(1e-9).log())
        loss[~mask] = 0
        loss = loss.sum(list(range(1, loss.ndim)))
        return loss

    @classmethod
    def dispatch(cls, mode: str, p, q, **kwargs):
        if mode == 'none':
            return 0.
        if mode == 'local':
            return cls.local(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparseall':
            return cls.local_sparseall(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparsetarget':
            return cls.local_sparsetarget(p, q, kwargs['mask'], kwargs.get('dim', -1))
        if isinstance(p, Tensor):
            p = cls.struct_constructor(p)
            q = cls.struct_constructor(q)
        if mode == 'marginal':
            return cls.marginal(p, q, kwargs['mask'])
        if mode == 'exact':
            return cls.exact(p, q)
        if mode.startswith('top'):
            return cls.topk(p, q, int(mode[3:]), False)
        raise KeyError

    @classmethod
    def register_struct_constructor(cls, func):
        cls.struct_constructor = func

class JS:
    @classmethod
    def dispatch(cls, mode: str, p, q, **kwargs):
        if mode == 'none':
            return 0.
        if mode == 'local':
            return KL.local(p, q, kwargs['mask'], kwargs.get('dim', -1)) + KL.local(q, p, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparseall':
            return KL.local_sparseall(p, q, kwargs['mask'], kwargs.get('dim', -1)) + KL.local_sparseall(q, p, kwargs['mask'], kwargs.get('dim', -1))
        if mode == 'local_sparsetarget':
            return KL.local_sparsetarget(p, q, kwargs['mask'], kwargs.get('dim', -1)) + KL.local_sparsetarget(q, p, kwargs['mask'], kwargs.get('dim', -1))
        if isinstance(p, Tensor):
            p = cls.struct_constructor(p)
            q = cls.struct_constructor(q)
        if mode == 'marginal':
            return KL.marginal(p, q, kwargs['mask']) + KL.marginal(q, p, kwargs['mask'])
        if mode == 'exact':
            return KL.exact(p, q) + KL.exact(q, p)
        if mode.startswith('top'):
            return KL.topk(p, q, int(mode[3:]), False) + KL.topk(q, p, int(mode[3:]), False)
        raise KeyError

    @staticmethod
    def register_struct_constructor(func):
        KL.struct_constructor = func

def normalize(d: Tensor, is_sentence_level, p='fro') -> Tensor:
    """
    d: [batch, max_len, hidden_size]
    is_sentence_level: True to norm on (1...). False to norm on (2...)
    emb_size: [n_emb], sum(emb_size) should be = hidden_size
    active_emb: List[n_emb], if False, set to 0. None=All True.
    """
    d = d.contiguous()
    shape = d.shape[1:] if is_sentence_level else d.shape[2:]
    d = d.view(d.shape[0], -1) if is_sentence_level else d.view(d.shape[0], d.shape[1], -1)

    d = d.clone()
    norm = torch.norm(d, p=p, dim=-1, keepdim=True)
    d.div_(norm + 1e-6)
    d.masked_fill_(norm < 1e-6, 0)

    d = d.view(*d.shape[:-1], *shape)
    return d


def scale(d: Tensor, size: int, scale_type: str, seq_len: Tensor) -> Tensor:
    if scale_type in ('token', 'sentence'):
        return size * d
    if scale_type == 'log_length':
        seq_len = seq_len.float().log()
    seq_len = seq_len.view(-1, *[1 for _ in range(d.ndim - 1)])
    return size * seq_len * d

def switch_dropout(module: torch.nn.Module, enable=True):
    if enable:
        for submodule in module.children():
            if isinstance(submodule, (torch.nn.modules.dropout._DropoutNd, SharedDropout, IndependentDropout)):
                if getattr(submodule, '_orig_p') is not None:
                    submodule.p = submodule._orig_p
                    submodule._orig_p = None
            elif isinstance(submodule, torch.nn.RNNBase):
                if getattr(submodule, '_orig_p') is not None:
                    submodule.dropout = submodule._orig_p
                    submodule._orig_p = None
    else:
        for submodule in module.children():
            if isinstance(submodule, (torch.nn.modules.dropout._DropoutNd, SharedDropout, IndependentDropout)):
                submodule._orig_p = submodule.p
            elif isinstance(submodule, torch.nn.RNNBase):
                submodule._orig_p = submodule.dropout


if __name__ == '__main__':
    p = DependencyCRF(torch.randn(1, 3,3), lens=torch.tensor([2]))
    q = DependencyCRF(torch.randn(1, 3,3), lens=torch.tensor([2]))
    print(CrossEntropy.exact(p, q))
    print(CrossEntropy.topk(p, q, 2, False))