import torch


def discretize_sigmoid(scores, deterministic: bool = True) -> torch.Tensor:
    if deterministic:
        discrete_scores = torch.ge(scores, 0.5)
    else:
        uniform_samples = torch.rand_like(scores)
        discrete_scores = torch.ge(scores, uniform_samples)

    return discrete_scores.float()


def discretize_softmax(
    scores: torch.Tensor, deterministic: bool = True, dim: int = 1
) -> torch.Tensor:
    if deterministic:
        discrete_scores = torch.argmax(scores, dim=dim)
        discrete_scores = torch.nn.functional.one_hot(
            discrete_scores, num_classes=scores.shape[dim]
        )
        discrete_scores = discrete_scores.transpose(dim, -1)
    else:
        discrete_scores = torch.nn.functional.gumbel_softmax(
            scores, tau=1.0, hard=True, dim=dim
        )

    return discrete_scores.float()


def make_mask_2d(lengths: torch.Tensor, expand_dim: int = None):
    """Create binary mask from lengths indicating which indices are padding"""
    # Make sure `lengths` is a 1d array
    assert len(lengths.shape) == 1

    max_length = torch.amax(lengths, dim=0).item()
    mask = torch.arange(max_length).expand(
        (lengths.shape[0], max_length)
    )  # Shape batch x timesteps
    mask = torch.ge(mask, lengths.unsqueeze(1))

    if expand_dim is not None:
        mask = mask.unsqueeze(2)
        mask = mask.expand((mask.shape[0], mask.shape[1], expand_dim))

    return mask


def make_mask_3d(source_lengths: torch.Tensor, target_lengths: torch.Tensor):
    """
    Make binary mask indicating which combinations of indices involve at least 1 padding element.
    Can be used to mask, for example, a batch attention matrix between 2 sequences
    """
    # Calculate binary masks for source and target
    # Then invert boolean values and convert to float (necessary for bmm later)
    source_mask = (~make_mask_2d(source_lengths)).float()
    target_mask = (~make_mask_2d(target_lengths)).float()

    # Add dummy dimensions for bmm
    source_mask = source_mask.unsqueeze(2)
    target_mask = target_mask.unsqueeze(1)

    # Calculate combinations by batch matrix multiplication
    mask = torch.bmm(source_mask, target_mask).bool()
    # Invert boolean values
    mask = torch.logical_not(mask)
    return mask
