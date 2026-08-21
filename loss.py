import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_softmax

def get_positive_expectation(p_samples, measure, average=True):
    log_2 = np.log(2.)
    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = np.log(2.)
    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    if average:
        return Eq.mean()
    else:
        return Eq

def global_global_contrast_loss(g1, g2, measure='JSD', tau=1, normalize=True):
    if normalize:
        g1 = F.normalize(g1, p=2, dim=1)
        g2 = F.normalize(g2, p=2, dim=1)
    scores = torch.matmul(g1, g2.t()) / tau  # [B, B]
    B = scores.size(0)
    pos_scores = scores.diag()  # [B]
    mask = ~torch.eye(B, dtype=torch.bool, device=scores.device)
    neg_scores = scores[mask]  # [B*(B-1)]

    pos_exp = get_positive_expectation(pos_scores, measure, average=False)
    neg_exp = get_negative_expectation(neg_scores, measure, average=False)

    loss = -(pos_exp.mean() + neg_exp.mean())
    return loss


def global_local_contrast_loss(global_reps, node_reps, batch, measure='JSD', tau=1, normalize=True):
    if normalize:
        global_reps = F.normalize(global_reps, p=2, dim=1)
        node_reps = F.normalize(node_reps, p=2, dim=1)
    scores = torch.matmul(global_reps, node_reps.t()) / tau  # [B, N]

    B = global_reps.size(0)
    N = node_reps.size(0)
    pos_mask = torch.zeros_like(scores, dtype=torch.bool)
    pos_mask[batch, torch.arange(N, device=scores.device)] = True

    pos_scores = scores[pos_mask]
    neg_scores = scores[~pos_mask]

    pos_exp = get_positive_expectation(pos_scores, measure, average=False)
    neg_exp = get_negative_expectation(neg_scores, measure, average=False)

    loss = -(pos_exp.mean() + neg_exp.mean())
    return loss


def diversity_representation_loss(views_reps):
    n_views = len(views_reps)
    if n_views < 2:
        return torch.tensor(0.0, device=views_reps[0].device)

    batch_size = views_reps[0].size(0)
    device = views_reps[0].device
    stacked = torch.stack(views_reps, dim=0)
    stacked = F.normalize(stacked, p=2, dim=2)

    sim = torch.einsum('kbd,lbd->klb', stacked, stacked)  # [n, n, B]

    mask = torch.triu(torch.ones(n_views, n_views, device=device, dtype=torch.bool), diagonal=1)
    pair_sims = sim[mask]  # [num_pairs, batch_size]
    loss = pair_sims.mean()

    return loss


def specificity_node_distribution_loss_fast(views_scores, batch):
    n_views = len(views_scores)
    if n_views < 2:
        return torch.tensor(0.0, device=views_scores[0].device)

    device = views_scores[0].device
    total_nodes = views_scores[0].size(0)

    probs_list = []
    for scores in views_scores:
        scores = scores.squeeze(-1)
        probs = scatter_softmax(scores / 0.1, batch, dim=0)
        probs_list.append(probs)
    probs = torch.stack(probs_list, dim=0)

    sorted_batch, sort_idx = batch.sort()
    sorted_probs = probs[:, sort_idx]

    num_nodes_per_graph = torch.bincount(sorted_batch)
    batch_size = len(num_nodes_per_graph)
    splits = num_nodes_per_graph.tolist()
    graph_probs_list = torch.split(sorted_probs, splits, dim=1)

    graph_probs_t = [p.t() for p in graph_probs_list]
    padded_probs = torch.nn.utils.rnn.pad_sequence(
        graph_probs_t, batch_first=True
    )

    padded_probs_norm = F.normalize(padded_probs, p=2, dim=1)


    sim = torch.bmm(padded_probs_norm.transpose(1, 2), padded_probs_norm)


    triu_mask = torch.triu(torch.ones(n_views, n_views, device=device, dtype=torch.bool), diagonal=1)
    loss_per_graph = sim[:, triu_mask].mean(dim=1)

    valid_mask = num_nodes_per_graph > 0
    loss = loss_per_graph[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=device)

    return loss


def perturb_scores(all_scores,
                   perturb_type='random_zero',
                   ratio=0.1,
                   gumble_tau=1.0,
                   noise_std=0.1):

    device = all_scores[0].device
    perturbed_scores = []

    if perturb_type =='none':
        return all_scores

    for scores in all_scores:
        if perturb_type == 'random_zero':
            mask = torch.rand_like(scores) < ratio
            scores[mask] = 0.0
        elif perturb_type == 'sign_flip':
            mask = torch.rand_like(scores) < ratio
            scores[mask] = -scores[mask]
        elif perturb_type == 'gumble':
            gumble = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10))
            scores = torch.sigmoid((scores + gumble) / gumble_tau)
        elif perturb_type == 'gaussian_noise':
            noise_std = noise_std * torch.std(scores)
            noise = torch.randn_like(scores) * noise_std
            scores = scores + noise
        else:
            raise ValueError(f"Unknown perturb_type: {perturb_type}")

        perturbed_scores.append(scores)

    return perturbed_scores