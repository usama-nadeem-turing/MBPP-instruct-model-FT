import torch
import torch.nn.functional as F
from swift.plugin.loss import register_loss_func

@register_loss_func("dft")  # use "custom_loss" if you prefer
def loss_scale_func(
    outputs,
    labels,
    loss_scale: float | None = None,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Dynamic Fine-Tuning loss = p_t (stop-grad) * CE

    Args
    ----
    outputs: model forward pass output with .logits [B, L, V]
    labels:  token ids [B, L] (use -100 for padding, same as HF)
    loss_scale: optional multiplicative factor applied at the end
    num_items_in_batch: optional divisor (e.g. micro-batch → global-batch)

    Returns
    -------
    torch.Tensor scalar loss
    """
    # Align logits / labels the same way ms-swift’s default CE does
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Drop padding/ignore_index positions
    mask = shift_labels.ne(-100)
    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]

    # −log pₜ  (standard cross-entropy, no reduction yet)
    ce = F.cross_entropy(shift_logits, shift_labels, reduction="none")

    # pₜ    (stop-grad so gradients don’t flow through the weight)
    prob_detached = torch.exp(-ce).detach()

    # One-line DFT: weight CE by detached probability
    loss = (prob_detached * ce).mean()

    # Optional scaling hooks that ms-swift may pass in
    if num_items_in_batch is not None and num_items_in_batch > 0:
        loss = loss / num_items_in_batch
    if loss_scale is not None:
        loss = loss * loss_scale

    return loss