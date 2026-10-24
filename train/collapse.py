"""Softmax collapse check (Prieto et al., "Grokking at the Edge of Numerical Stability", ICLR 2025).

After a model fits its training data, cross-entropy keeps pushing the right token's logit away from the rest. Once the
gap is large enough, the softmax gives exactly 1.0 for the right token in floating point: the loss and the gradient
for that token are then exactly 0 and it stops contributing to learning. In float32 that happens for a logit gap of
about 16-17 (exp(-gap) below float32 resolution near 1.0); fp16 / bf16 have far fewer bits (threshold ~7 / ~5).

collapse_stats(...) runs the model on TRAINING examples with the answer given (teacher forcing, one forward pass,
exactly the training computation) and reports, over every answer token:
  p_mean, loss_mean           mean probability of the right token / mean cross-entropy
  gap_median, gap_p90         logit gap: right token minus the best wrong token
  zero_grad_fp32              share of answer tokens whose float32 softmax gives exactly 1.0 (zero gradient)
  zero_grad_bf16 / fp16       the same if the softmax were computed in bf16 / fp16
  near_fp32                   share with gap > 15 (close to the float32 threshold)
  logit_norm, hidden_norm     mean L2 norm of the logit vector / of the last hidden state at answer positions
"""

from __future__ import annotations

import torch


@torch.no_grad()
def collapse_stats(model, tok, examples: list[dict], device, amp, bsz: int = 8) -> dict:
    """examples: dicts with "ids" and "labels" (-100 on prompt tokens), as used in training."""
    from train.run import collate
    was = model.training
    model.eval()
    gaps, ps, losses, zf32, zbf, zf16, lnorm, hnorm = [], [], [], [], [], [], [], []
    for s in range(0, len(examples), bsz):
        b = collate(examples[s:s + bsz], tok.pad_token_id, device)
        with amp():
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        logits = out.logits[:, :-1]
        hid = out.hidden_states[-1][:, :-1]
        tgt = b["labels"][:, 1:]
        m = tgt != -100
        sel, t_all = logits[m], tgt[m]                             # (tokens, vocab) in the model's dtype
        for r in range(0, sel.shape[0], 64):                      # 64 answer tokens at a time (GPU memory)
            lg, t = sel[r:r + 64].float(), t_all[r:r + 64]        # float32, as the loss sees them
            right = lg.gather(1, t[:, None]).squeeze(1)
            other = lg.scatter(1, t[:, None], float("-inf"))
            gaps.append(right - other.max(1).values)
            del other
            losses.append(-torch.log_softmax(lg, -1).gather(1, t[:, None]).squeeze(1))
            p32 = torch.softmax(lg, -1).gather(1, t[:, None]).squeeze(1)
            ps.append(p32)
            zf32.append(p32 == 1.0)
            zbf.append(torch.softmax(lg.to(torch.bfloat16), -1).gather(1, t[:, None]).squeeze(1) == 1.0)
            zf16.append(torch.softmax(lg.to(torch.float16), -1).gather(1, t[:, None]).squeeze(1) == 1.0)
            lnorm.append(lg.norm(dim=-1))
        hnorm.append(hid[m].float().norm(dim=-1))
    cat = lambda xs: torch.cat(xs).float()  # noqa: E731
    g, p = cat(gaps), cat(ps)
    if was:
        model.train()
    return {"tokens": int(g.numel()), "p_mean": p.mean().item(), "loss_mean": cat(losses).mean().item(),
            "gap_median": g.median().item(), "gap_p90": g.quantile(0.9).item(),
            "zero_grad_fp32": cat(zf32).mean().item(), "zero_grad_bf16": cat(zbf).mean().item(),
            "zero_grad_fp16": cat(zf16).mean().item(), "near_fp32": (g > 15).float().mean().item(),
            "logit_norm": cat(lnorm).mean().item(), "hidden_norm": cat(hnorm).mean().item()}
