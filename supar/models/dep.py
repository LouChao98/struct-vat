# -*- coding: utf-8 -*-

from functools import partial
import torch
import torch.nn as nn
from torch.autograd import grad
from supar.models.model import Model
from supar.modules import MLP, Biaffine, Triaffine
from supar.modules.contrastive import CrossEntropy, KL, JS, normalize, scale, switch_dropout
from supar.structs import (Dependency2oCRF, DependencyCRF, DependencyLBP,
                           DependencyMFVI, MatrixTree)
from supar.utils import Config
from supar.utils.common import MIN
from supar.utils.transform import CoNLL


class BiaffineDependencyModel(Model):
    r"""
    The implementation of Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 # constrastive
                 contrastive_algo = 'vat',  # vat, vat_sym(kl(p,q)+kl(q,p))
                 contrastive_distance = 'ce',  # ce, kl
                 # vat
                 vat_mode = 'exact',  # exact, top-n, local
                 vat_distribution = 'gaussian',
                 vat_xi = 0.2, # perturb size when searching (std for guassian)
                 vat_eps = 0.05,   # perturb size when outputing (std for guassian)
                 vat_ip = 1,  # iteration
                 vat_scale_type = 'token',  # token sentence log
                 vat_detach_p = False,
                 vat_interpolation = 0.5,  # 0 for full arc
                 vat_1 = False,  # this will disable interpolation
                 vat_filter_mode = 'none',  # none, same, samller
                 vat_temperature_p = 1.,
                 vat_temperature_q = 1.,
                 # rdrop
                 rdrop_mode = 'exact',
                 rdrop_interpolation = 0.5,
                 rdrop_temperature_p = 1.,
                 rdrop_temperature_q = 1.,
                 **kwargs): 
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.ln = None

    def forward(self, words, feats=None, embed=None, mask=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """
        if mask is None:
            mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        x = self.encode(words, feats, embed=embed)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def constrastive_loss(self, words, feats=None, prev=None):
        if self.args.contrastive_algo == 'vat':
            return self.vat_loss(words, feats, prev)
        if self.args.contrastive_algo == 'rdrop':
            return self.rdrop_loss(words, feats, prev)

    def get_distance(self):
        if self.args.contrastive_distance == 'ce':
            return CrossEntropy
        elif self.args.contrastive_distance == 'kl':
            return KL
        elif self.args.contrastive_distance == 'js':
            return JS

    def vat_loss(self, words, feats=None, prev=None):

        # prepare emb and initial perturb
        switch_dropout(self, False)
        x = self.embed(words, feats) if self.args.encoder == 'lstm' else self.encoder(words, feats)
        p_arc, p_label = self(words, feats, embed=x)
        if self.args.vat_detach_p and self.args.contrastive_distance:
            p_arc, p_label = p_arc.detach(), p_label.detach()
        switch_dropout(self, True)

        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        length = mask.sum(1) - 1
        adv_d = torch.rand_like(x)
        if self.args.vat_distribution == 'gaussian':
            if self.ln is None: 
                self.ln = nn.LayerNorm(adv_d.shape[-1], elementwise_affine=False)
        else:
            adv_d -= 0.5
        if self.args.vat_distribution != 'gaussian':
            adv_d[~mask] = 0.

        # setup struct
        DISTANCE = self.get_distance()
        DISTANCE.register_struct_constructor(partial(DependencyCRF, lens=length))

        # vat
        for _ in range(self.args.vat_ip):
            adv_d = normalize(adv_d, self.args.vat_scale_type != 'token') if self.ln is None else self.ln(adv_d)
            adv_d = scale(adv_d, self.args.vat_xi, self.args.vat_scale_type, length)
            adv_d = adv_d.requires_grad_()
            q_arc, q_label = self(words, feats, embed=x + adv_d, mask=mask)
            loss = self._vat_loss(p_arc, p_label, q_arc, q_label, mask)
            adv_d = grad(loss, adv_d)[0].detach()
            if self.args.vat_distribution != 'gaussian':
                adv_d[~mask] = 0.

        adv_d = normalize(adv_d, self.args.vat_scale_type != 'token') if self.ln is None else self.ln(adv_d)
        adv_d = scale(adv_d, self.args.vat_eps, self.args.vat_scale_type, length)
        q_arc, q_label = self(words, feats, embed=x + adv_d, mask=mask)

        loss = self._vat_loss(p_arc, p_label, q_arc, q_label, mask)
        return loss / mask.sum()

    def _vat_loss(self, p_arc, p_label, q_arc, q_label, mask):
        arc_loss, label_loss = 0, 0
        pred_p = None
        intplot = self.args.vat_interpolation
        if self.args.vat_temperature_p != 1.:
            p_arc = p_arc / self.args.vat_temperature_p
            p_label = p_label / self.args.vat_temperature_p
        if self.args.vat_temperature_q != 1.:
            q_arc = q_arc / self.args.vat_temperature_q
            q_label = q_label / self.args.vat_temperature_q

        DISTANCE = self.get_distance()
        if intplot < 1 or self.args.vat_1:
            arc_loss = DISTANCE.dispatch(self.args.vat_mode, p_arc, q_arc, mask=mask).clamp(0)
        if intplot > 0 or self.args.vat_1:
            if pred_p is None:
                pred_p = p_arc.argmax(-1)
            _pred_p = pred_p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, p_label.shape[-1])
            _p_label = p_label.gather(2, _pred_p).squeeze(2)
            _q_label = q_label.gather(2, _pred_p).squeeze(2)
            label_loss = DISTANCE.dispatch('local', _p_label, _q_label, mask=mask).clamp(0) / p_label.shape[-1]
        if self.args.vat_1:
            loss: torch.Tensor = arc_loss / arc_loss.detach().clamp(1) + label_loss / label_loss.detach().clamp(1)
        else:
            loss: torch.Tensor = (1 - intplot) * arc_loss + intplot * label_loss
        if self.args.vat_filter_mode == 'same':
            pred_p, pred_q = p_arc.argmax(-1) if pred_p is None else pred_p, q_arc.argmax(-1)
            mask = (pred_p == pred_q).all(-1)
            loss.masked_fill_(mask, 0)
        if self.args.vat_filter_mode == 'smaller':
            pred_p = p_arc.max(-1)[0].sum(-1)
            pred_q = q_arc.max(-1)[0].sum(-1)
            mask = pred_p <= pred_q
            loss.masked_fill_(mask, 0)
        return loss.sum()

    def rdrop_loss(self, words, feats=None, prev=None):
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        length = mask.sum(1) - 1
        p_arc, p_label = self(words, feats)
        q_arc, q_label = self(words, feats)
        
        if self.args.rdrop_temperature_p != 1.:
            p_arc = p_arc / self.args.rdrop_temperature_p
            p_label = p_label / self.args.rdrop_temperature_p
        if self.args.rdrop_temperature_q != 1.:
            q_arc = q_arc / self.args.rdrop_temperature_q
            q_label = q_label / self.args.rdrop_temperature_q

        intplot = self.args.rdrop_interpolation
        
        DISTANCE = self.get_distance()
        DISTANCE.register_struct_constructor(partial(DependencyCRF, lens=length))
        arc_loss, label_loss = 0, 0
        if intplot < 1:
            arc_loss = DISTANCE.dispatch(self.args.rdrop_mode, p_arc, q_arc, mask=mask).clamp(0)
        if intplot > 0:
            pred_p = p_arc.argmax(-1)
            _pred_p = pred_p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, p_label.shape[-1])
            _p_label = p_label.gather(2, _pred_p).squeeze(2)
            _q_label = q_label.gather(2, _pred_p).squeeze(2)
            label_loss = DISTANCE.dispatch('local', _p_label, _q_label, mask=mask).clamp(0) / p_label.shape[-1]
        loss: torch.Tensor = (1 - intplot) * arc_loss + intplot * label_loss
        return loss.sum()

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class CRFDependencyModel(BiaffineDependencyModel):
    r"""
    The implementation of first-order CRF Dependency Parser
    :cite:`zhang-etal-2020-efficient,ma-hovy-2017-neural,koo-etal-2007-structured`).

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
        proj (bool):
            If ``True``, takes :class:`DependencyCRF` as inference layer, :class:`MatrixTree` otherwise.
            Default: ``True``.
    """

    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=True, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        CRF = DependencyCRF if self.args.proj else MatrixTree
        arc_dist = CRF(s_arc, mask.sum(-1))
        arc_loss = -arc_dist.log_prob(arcs, partial=partial).sum() / mask.sum()
        arc_probs = arc_dist.marginals if mbr else s_arc
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs


class CRF2oDependencyModel(BiaffineDependencyModel):
    r"""
    The implementation of second-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_sib_mlp (int):
            Sibling MLP size. Default: 100.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_sib_mlp=100,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.sib_mlp_s = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=n_sib_mlp, scale=scale, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None, embed=None, mask=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible arcs (``[batch_size, seq_len, seq_len]``),
                dependent-head-sibling triples (``[batch_size, seq_len, seq_len, seq_len]``) and
                all possible labels on each arc (``[batch_size, seq_len, seq_len, n_labels]``).
        """

        if mask is None:
            mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        x = self.encode(words, feats, embed=embed)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        sib_s = self.sib_mlp_s(x)
        sib_d = self.sib_mlp_d(x)
        sib_h = self.sib_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask, mbr=True, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            sibs (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard siblings.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        arc_dist = Dependency2oCRF((s_arc, s_sib), mask.sum(-1))
        arc_loss = -arc_dist.log_prob((arcs, sibs), partial=partial).sum() / mask.sum()
        if mbr:
            s_arc, s_sib = arc_dist.marginals
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, s_arc, s_sib
    
    def vat_loss(self, words, feats=None, prev=None):

        # prepare emb and initial perturb
        switch_dropout(self, False)
        x = self.embed(words, feats) if self.args.encoder == 'lstm' else self.encoder(words, feats)
        p_arc, p_sib, p_label = self(words, feats, embed=x)
        if self.args.vat_detach_p and self.args.contrastive_distance:
            p_arc, p_sib, p_label = p_arc.detach(), p_sib.detach(), p_label.detach()
        switch_dropout(self, True)

        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        length = mask.sum(1) - 1
        adv_d = torch.rand_like(x)
        if self.args.vat_distribution == 'gaussian':
            if self.ln is None: 
                self.ln = nn.LayerNorm(adv_d.shape[-1], elementwise_affine=False)
        else:
            adv_d -= 0.5
        if self.args.vat_distribution != 'gaussian':
            adv_d[~mask] = 0.

        # setup struct
        DISTANCE = self.get_distance()
        DISTANCE.register_struct_constructor(partial(Dependency2oCRF, lens=length))

        # vat
        for _ in range(self.args.vat_ip):
            adv_d = normalize(adv_d, self.args.vat_scale_type != 'token') if self.ln is None else self.ln(adv_d)
            adv_d = scale(adv_d, self.args.vat_xi, self.args.vat_scale_type, length)
            adv_d = adv_d.requires_grad_()
            q_arc, q_sib, q_label = self(words, feats, embed=x + adv_d, mask=mask)
            loss = self._vat_loss(p_arc, p_sib, p_label, q_arc, q_sib, q_label, mask)
            adv_d = grad(loss, adv_d)[0].detach()
            if self.args.vat_distribution != 'gaussian':
                adv_d[~mask] = 0.

        adv_d = normalize(adv_d, self.args.vat_scale_type != 'token') if self.ln is None else self.ln(adv_d)
        adv_d = scale(adv_d, self.args.vat_eps, self.args.vat_scale_type, length)
        q_arc, q_sib, q_label = self(words, feats, embed=x + adv_d, mask=mask)

        loss = self._vat_loss(p_arc, p_sib, p_label, q_arc, q_sib, q_label, mask)
        return loss / mask.sum()

    def _vat_loss(self, p_arc, p_sib, p_label, q_arc, q_sib, q_label, mask):
        arc_loss, label_loss = 0, 0
        pred_p = None
        intplot = self.args.vat_interpolation
        if self.args.vat_temperature_p != 1.:
            p_arc = p_arc / self.args.vat_temperature_p
            p_sib = p_sib / self.args.vat_temperature_p
            p_label = p_label / self.args.vat_temperature_p
        if self.args.vat_temperature_q != 1.:
            q_arc = q_arc / self.args.vat_temperature_q
            q_sib = q_sib / self.args.vat_temperature_q
            q_label = q_label / self.args.vat_temperature_q
      
        DISTANCE = self.get_distance()
        if intplot < 1 or self.args.vat_1:
            arc_loss = DISTANCE.dispatch(self.args.vat_mode, 
                p_arc if self.args.vat_mode in ('local', 'local_sparseall', 'local_sparsetarget') else (p_arc, p_sib), 
                q_arc if self.args.vat_mode in ('local', 'local_sparseall', 'local_sparsetarget') else (q_arc, q_sib), 
                mask=mask).clamp(0)
        if intplot > 0 or self.args.vat_1:
            if pred_p is None:
                pred_p = p_arc.argmax(-1)
            _pred_p = pred_p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, p_label.shape[-1])
            _p_label = p_label.gather(2, _pred_p).squeeze(2)
            _q_label = q_label.gather(2, _pred_p).squeeze(2)
            label_loss = DISTANCE.dispatch('local', _p_label, _q_label, mask=mask).clamp(0) / p_label.shape[-1]
        if self.args.vat_1:
            loss: torch.Tensor = arc_loss / arc_loss.detach().clamp(1) + label_loss / label_loss.detach().clamp(1)
        else:
            loss: torch.Tensor = (1 - intplot) * arc_loss + intplot * label_loss
        if self.args.vat_filter_mode == 'same':
            pred_p, pred_q = p_arc.argmax(-1) if pred_p is None else pred_p, q_arc.argmax(-1)
            mask = (pred_p == pred_q).all(-1)
            loss.masked_fill_(mask, 0)
        if self.args.vat_filter_mode == 'smaller':
            pred_p = p_arc.max(-1)[0].sum(-1)
            pred_q = q_arc.max(-1)[0].sum(-1)
            mask = pred_p <= pred_q
            loss.masked_fill_(mask, 0)
        return loss.sum()

    def decode(self, s_arc, s_sib, s_rel, mask, tree=False, mbr=True, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            if proj:
                arc_preds[bad] = Dependency2oCRF((s_arc[bad], s_sib[bad]), mask[bad].sum(-1)).argmax
            else:
                arc_preds[bad] = MatrixTree(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class VIDependencyModel(BiaffineDependencyModel):
    r"""
    The implementation of Dependency Parser using Variational Inference :cite:`wang-tu-2020-second`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_sib_mlp (int):
            Binary factor MLP size. Default: 100.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        inference (str):
            Approximate inference methods. Default: ``mfvi``.
        max_iter (int):
            Max iteration times for inference. Default: 3.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_sib_mlp=100,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 inference='mfvi',
                 max_iter=3,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.sib_mlp_s = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=n_sib_mlp, scale=scale, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.inference = (DependencyMFVI if inference == 'mfvi' else DependencyLBP)(max_iter)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible arcs (``[batch_size, seq_len, seq_len]``),
                dependent-head-sibling triples (``[batch_size, seq_len, seq_len, seq_len]``) and
                all possible labels on each arc (``[batch_size, seq_len, seq_len, n_labels]``).
        """

        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        sib_s = self.sib_mlp_s(x)
        sib_d = self.sib_mlp_d(x)
        sib_h = self.sib_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, rels, mask):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        arc_loss, marginals = self.inference((s_arc, s_sib), mask, arcs)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, marginals

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
