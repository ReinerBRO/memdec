import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np
from loguru import logger
import os

EMBED_PAD_NUM = -10000
KNN_EPS = 1e-12

def interpolate(knn_log_probs, lm_log_probs, lmbda=0.25):
    interpolated = torch.logaddexp(
        lm_log_probs + np.log(1 - lmbda), 
        knn_log_probs + np.log(lmbda))

    return interpolated

def _get_shifted_logits_and_labels(logits, batch):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask]
    shift_labels = shift_labels[nonpad_mask]

    return shift_logits, shift_labels

def _prepare_sparse_knn(batch, knn_prob):
    if "knn_token_ids" not in batch or "knn_token_mask" not in batch:
        return None, None, None

    knn_token_ids = batch["knn_token_ids"]
    knn_token_mask = batch["knn_token_mask"]
    knn_prob = torch.where(knn_token_mask, knn_prob, torch.zeros_like(knn_prob))
    knn_prob = knn_prob / knn_prob.sum(dim=-1, keepdim=True).clamp_min(KNN_EPS)

    return knn_token_ids, knn_token_mask, knn_prob

def _sparse_target_log_probs(shift_logits, shift_labels, knn_token_ids, knn_prob):
    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    lm_target_log_probs = lm_log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    target_mask = knn_token_ids.eq(shift_labels.unsqueeze(-1))
    knn_target_probs = torch.where(target_mask, knn_prob, torch.zeros_like(knn_prob)).sum(dim=-1)
    knn_target_log_probs = torch.where(
        knn_target_probs > 0,
        knn_target_probs.log(),
        torch.full_like(knn_target_probs, EMBED_PAD_NUM, dtype=lm_target_log_probs.dtype),
    )

    return lm_target_log_probs, knn_target_log_probs

def _sparse_kl_loss(shift_logits, knn_token_ids, knn_prob):
    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_lm_log_probs = lm_log_probs.gather(dim=-1, index=knn_token_ids)
    knn_log_probs = torch.where(knn_prob > 0, knn_prob.log(), torch.zeros_like(knn_prob))
    per_token_kl = (knn_prob * (knn_log_probs - gathered_lm_log_probs)).sum(dim=-1)

    return per_token_kl.mean()

def kl_loss_evaluate(logits, batch, tokenizer, args, knn_label, knn_prob):
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, batch)
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"

    knn_token_ids, _, sparse_knn_prob = _prepare_sparse_knn(batch, knn_prob)
    if sparse_knn_prob is not None:
        assert shift_labels.shape[0] == sparse_knn_prob.shape[0], f"shift_labels.shape[0] = {shift_labels.shape[0]}, sparse_knn_prob.shape[0] = {sparse_knn_prob.shape[0]}"
        assert torch.allclose(
            sparse_knn_prob.sum(dim=-1),
            torch.ones_like(sparse_knn_prob.sum(dim=-1)),
            atol=1e-4,
        ), f"sparse_knn_prob does not sum to 1"

        lm_target_log_probs, knn_target_log_probs = _sparse_target_log_probs(
            shift_logits, shift_labels, knn_token_ids, sparse_knn_prob
        )
        interpolate_target_log_probs = interpolate(knn_target_log_probs, lm_target_log_probs, lmbda=args.lmbda)
        nll_loss = -interpolate_target_log_probs.sum()
        lm_loss = -lm_target_log_probs.sum()
    else:
        label_probs = knn_prob / knn_prob.sum(dim=-1, keepdim=True).clamp_min(KNN_EPS)
        assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
        assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"

        label_log_probs = label_probs.log()
        label_log_probs = torch.nan_to_num(label_log_probs, nan=None, neginf=EMBED_PAD_NUM)
        lm_log_probs = F.log_softmax(shift_logits, dim=-1)
        interpolate_log_probs = interpolate(label_log_probs, lm_log_probs, lmbda=args.lmbda)
        nll_loss = F.nll_loss(interpolate_log_probs, shift_labels, reduction='sum')
        lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')

    token_num = shift_labels.shape[0]
    
    return nll_loss, lm_loss, token_num

def kl_loss_token(logits, batch, tokenizer, args, knn_label, knn_prob, alpha=0.5):
    shift_logits, shift_labels = _get_shifted_logits_and_labels(logits, batch)
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"

    knn_token_ids, _, sparse_knn_prob = _prepare_sparse_knn(batch, knn_prob)
    if sparse_knn_prob is not None:
        assert shift_labels.shape[0] == sparse_knn_prob.shape[0], f"shift_labels.shape[0] = {shift_labels.shape[0]}, sparse_knn_prob.shape[0] = {sparse_knn_prob.shape[0]}"
        assert torch.allclose(
            sparse_knn_prob.sum(dim=-1),
            torch.ones_like(sparse_knn_prob.sum(dim=-1)),
            atol=1e-4,
        ), f"sparse_knn_prob does not sum to 1"
        kl_loss = _sparse_kl_loss(shift_logits, knn_token_ids, sparse_knn_prob)
    else:
        label_probs = knn_prob / knn_prob.sum(dim=-1, keepdim=True).clamp_min(KNN_EPS)
        assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
        assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"
        kl_loss = F.kl_div(F.log_softmax(shift_logits, dim=-1), label_probs, reduction='batchmean')
    
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits, shift_labels)
    
    total_loss = alpha * kl_loss + (1 - alpha) * lm_loss
    
    logger.info(f"KL loss: {kl_loss} LM loss: {lm_loss} Total loss: {total_loss}")
    
    return total_loss, kl_loss, lm_loss
