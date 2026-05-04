"""
Soft labeling loss for numerical token prediction.

Implements the method from:
"Enhancing Numerical Prediction of MLLMs with Soft Labeling" (ICCV 2025, Wang et al.)

Key equation: q^{SL}(t) = (1 - eta) * delta(t) + eta * psi(t)
where psi is a distance-aware distribution over digit tokens {0,...,9}.
Supported distributions: triangular, binomial, poisson, uniform.

Combined loss:
  L = (1 / (N_r + N_n)) * [ sum_regular CE(hard) + lambda * sum_numerical CE(soft) ]
"""

from math import comb, exp, factorial

import torch
import torch.nn.functional as F

from geochat.constants import IGNORE_INDEX


def build_digit_token_ids(tokenizer):
    """
    Extract the token IDs for digits '0' through '9' from the tokenizer.

    Uses convert_tokens_to_ids() to directly look up each digit character
    in the vocabulary, bypassing the SentencePiece encoding pipeline which
    would prepend a prefix space token (e.g., '▁0' -> [29871, 29900]).

    Args:
        tokenizer: A HuggingFace PreTrainedTokenizer instance.

    Returns:
        List[int]: A list of 10 token IDs, one per digit 0-9.
    """
    digit_ids = []
    for d in range(10):
        # Directly look up the token ID for the single character in vocab,
        # avoiding SentencePiece's automatic prefix-space behavior.
        tid = tokenizer.convert_tokens_to_ids(str(d))
        assert tid != tokenizer.unk_token_id, (
            f"Digit '{d}' mapped to UNK token (ID {tid}). "
            f"The tokenizer vocabulary does not contain a dedicated token for '{d}'."
        )
        digit_ids.append(tid)

    # Sanity check: all 10 digits should map to distinct token IDs
    assert len(set(digit_ids)) == 10, (
        f"Digit token IDs are not all unique: {digit_ids}. "
        f"This suggests a tokenizer vocabulary issue."
    )
    return digit_ids


def build_triangular_soft_matrix(digit_token_ids, vocab_size, eta=0.08):
    """
    Pre-compute a [10, vocab_size] soft label matrix using a triangular distribution.

    The triangular distribution assigns weight proportional to:
        psi(k | t) = max(0, 1 - |t - k| / 9) / Z
    where t is the target digit, k iterates over {0,...,9}, and Z normalizes.

    For each target digit t, the soft label row is:
        q[digit_token_ids[t]] = (1 - eta) + eta * psi(t | t)   (target position)
        q[digit_token_ids[k]] = eta * psi(k | t)               (other digits)
        q[other positions]    = 0                                (non-digit vocab)

    Args:
        digit_token_ids: List of 10 token IDs for digits 0-9.
        vocab_size: Total vocabulary size.
        eta: Mixing coefficient in [0, 1]. Default 0.08 (paper recommendation).

    Returns:
        Tensor: Shape [10, vocab_size]. Row i = soft target when true digit is i.
    """
    soft_matrix = torch.zeros(10, vocab_size)

    for t in range(10):
        # Compute triangular weights: closer digits get higher weight
        weights = torch.zeros(10)
        for k in range(10):
            dist = abs(t - k)
            weights[k] = max(0.0, 1.0 - dist / 9.0)
        # Normalize to a valid probability distribution
        weights = weights / weights.sum()

        # Build the full-vocab soft label row
        row = torch.zeros(vocab_size)
        # Hard label component: (1 - eta) on the true digit
        row[digit_token_ids[t]] = 1.0 - eta
        # Soft distribution component: eta * psi(k|t) on each digit
        for k in range(10):
            row[digit_token_ids[k]] += eta * weights[k].item()

        soft_matrix[t] = row

    return soft_matrix


def build_binomial_soft_matrix(digit_token_ids, vocab_size, eta=0.05):
    """
    Pre-compute a [10, vocab_size] soft label matrix using a binomial distribution.

    The binomial distribution:
        psi(k | t) = C(9, k) * (t/9)^k * (1 - t/9)^(9-k)
    where t is the target digit, k iterates over {0,...,9}.

    Args:
        digit_token_ids: List of 10 token IDs for digits 0-9.
        vocab_size: Total vocabulary size.
        eta: Mixing coefficient in [0, 1]. Default 0.05 (paper recommendation).

    Returns:
        Tensor: Shape [10, vocab_size]. Row i = soft target when true digit is i.
    """
    soft_matrix = torch.zeros(10, vocab_size)

    for t in range(10):
        # Binomial parameter p = t/9, clamped to avoid degenerate cases
        p = t / 9.0
        p = max(p, 1e-6)
        p = min(p, 1.0 - 1e-6)

        weights = torch.zeros(10)
        for k in range(10):
            weights[k] = comb(9, k) * (p ** k) * ((1.0 - p) ** (9 - k))
        # Renormalize for numerical safety
        weights = weights / weights.sum()

        row = torch.zeros(vocab_size)
        row[digit_token_ids[t]] = 1.0 - eta
        for k in range(10):
            row[digit_token_ids[k]] += eta * weights[k].item()

        soft_matrix[t] = row

    return soft_matrix


def build_poisson_soft_matrix(digit_token_ids, vocab_size, eta=0.02):
    """
    Pre-compute a [10, vocab_size] soft label matrix using a Poisson distribution.

    Following Beckham & Pal (2017), for target digit t the distribution is
    Poisson with rate lambda_t = max(t, 0.5) so the mode stays near t and the
    degenerate t=0 case does not collapse:
        psi(k | t) = (lambda_t^k * exp(-lambda_t)) / k!,  k in {0,...,9}
    After computing psi for k=0..9 we renormalise so the sum equals 1.

    Args:
        digit_token_ids: List of 10 token IDs for digits 0-9.
        vocab_size: Total vocabulary size.
        eta: Mixing coefficient in [0, 1]. Default 0.02 (paper recommendation).

    Returns:
        Tensor: Shape [10, vocab_size]. Row i = soft target when true digit is i.
    """
    soft_matrix = torch.zeros(10, vocab_size)

    for t in range(10):
        lam = max(float(t), 0.5)  # avoid lambda=0 which gives degenerate PMF
        weights = torch.zeros(10)
        for k in range(10):
            weights[k] = (lam ** k) * exp(-lam) / factorial(k)
        # Renormalise: truncating the PMF at k=9 loses mass for large lam
        weights = weights / weights.sum()

        row = torch.zeros(vocab_size)
        row[digit_token_ids[t]] = 1.0 - eta
        for k in range(10):
            row[digit_token_ids[k]] += eta * weights[k].item()

        soft_matrix[t] = row

    return soft_matrix


def build_uniform_soft_matrix(digit_token_ids, vocab_size, eta=0.05):
    """
    Pre-compute a [10, vocab_size] soft label matrix using a uniform distribution
    over the 10 digit tokens. This is equivalent to classical label smoothing
    restricted to the digit vocabulary, used in the paper as a *negative*
    control (Table 4: performs worst because it discards distance information).

        psi(k | t) = 1 / 10  for all k in {0,...,9}

    Args:
        digit_token_ids: List of 10 token IDs for digits 0-9.
        vocab_size: Total vocabulary size.
        eta: Mixing coefficient in [0, 1]. Default 0.05 (paper recommendation).

    Returns:
        Tensor: Shape [10, vocab_size]. Row i = soft target when true digit is i.
    """
    soft_matrix = torch.zeros(10, vocab_size)
    uniform_weight = 1.0 / 10.0

    for t in range(10):
        row = torch.zeros(vocab_size)
        row[digit_token_ids[t]] = 1.0 - eta
        for k in range(10):
            row[digit_token_ids[k]] += eta * uniform_weight
        soft_matrix[t] = row

    return soft_matrix


def compute_soft_label_loss(
    shift_logits,
    shift_labels,
    soft_label_matrix,
    digit_token_ids,
    lambda_weight=2.0,
):
    """
    Combined loss: standard CE for regular tokens + lambda-weighted soft CE for digit tokens.

    L = (1 / (N_r + N_n)) * [ sum_regular CE(hard) + lambda * sum_numerical CE(soft) ]

    Args:
        shift_logits: Tensor [N, vocab_size], flattened predicted logits.
        shift_labels: Tensor [N], flattened target token IDs (with IGNORE_INDEX for masked).
        soft_label_matrix: Tensor [10, vocab_size], pre-computed soft label distributions.
        digit_token_ids: List of 10 token IDs for digits 0-9.
        lambda_weight: Balancing weight for numerical token loss. Default 2.0.

    Returns:
        Scalar loss tensor.
    """
    # ---- Step 1: Identify valid, digit, and regular token positions ----
    valid_mask = shift_labels != IGNORE_INDEX  # [N]

    if valid_mask.sum() == 0:
        # No valid tokens in this batch (all masked) - return zero loss with grad
        return shift_logits.sum() * 0.0

    # Build a boolean mask for digit token positions
    is_digit = torch.zeros_like(shift_labels, dtype=torch.bool)
    for tid in digit_token_ids:
        is_digit |= (shift_labels == tid)

    digit_mask = valid_mask & is_digit       # [N]
    regular_mask = valid_mask & (~is_digit)   # [N]

    N_r = regular_mask.sum()
    N_n = digit_mask.sum()
    total = N_r + N_n

    if total == 0:
        return shift_logits.sum() * 0.0

    loss = torch.tensor(0.0, device=shift_logits.device, dtype=shift_logits.dtype)

    # ---- Step 2: Standard CE on regular (non-digit) tokens ----
    if N_r > 0:
        regular_logits = shift_logits[regular_mask]   # [N_r, vocab_size]
        regular_labels = shift_labels[regular_mask]    # [N_r]
        ce_regular = F.cross_entropy(regular_logits, regular_labels, reduction='sum')
        loss = loss + ce_regular

    # ---- Step 3: Soft CE on digit tokens ----
    if N_n > 0:
        digit_logits = shift_logits[digit_mask]    # [N_n, vocab_size]
        digit_labels = shift_labels[digit_mask]    # [N_n]

        # Map each digit token ID to its digit index (0-9)
        digit_indices = torch.zeros_like(digit_labels)
        for idx, tid in enumerate(digit_token_ids):
            digit_indices[digit_labels == tid] = idx

        # Gather the pre-computed soft label rows: [N_n, vocab_size]
        soft_targets = soft_label_matrix[digit_indices]

        # Soft cross-entropy: L = -sum( q_i * log_softmax(logits)_i )
        log_probs = F.log_softmax(digit_logits, dim=-1)  # [N_n, vocab_size]
        ce_soft = -(soft_targets * log_probs).sum(dim=-1).sum()  # scalar

        loss = loss + lambda_weight * ce_soft

    # ---- Step 4: Normalize by total valid token count ----
    loss = loss / total.float()

    return loss
