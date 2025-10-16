import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input + eps)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = cross_entropy_loss(input, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


class ConfidentialIntervalLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, min_value, max_value):
       
        mask = torch.where((pred <= max_value) & (pred >= min_value), target, pred)
        loss = cross_entropy_loss(mask, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        
        return loss

class ConfidentialIntervalLoss_CLAM(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, min_value, max_value):
        mask = torch.where((pred <= max_value) & (pred >= min_value), target, pred)
        loss = cross_entropy_loss(mask, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, task_name='cell_type'):
        with torch.no_grad():
            results_dict = model(x)
            pred = F.softmax(results_dict['cell_type_logits'], dim=-1)

        # print(pred.shape)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                # pred_hat = model(x + self.xi * d)
                results_dict = model(x + self.xi * d)
                pred_hat = results_dict['cell_type_logits']

                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            # pred_hat = model(x + r_adv)
            results_dict = model(x + r_adv)
            pred_hat = results_dict['cell_type_logits']
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")
        # print(lds)
        # exit()
        return lds


class VATLoss_CLAM(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_CLAM, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, task_name='cell_type'):
        with torch.no_grad():
            # logits, results_dict = model(x)
            logits, Y_prob, Y_hat, _, results_dict = model(x, instance_eval=False)
            pred = F.softmax(results_dict[task_name + '_logits'], dim=-1)

        # print(pred.shape)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                # pred_hat = model(x + self.xi * d)
                # logits, results_dict = model(x + self.xi * d)
                logits, Y_prob, Y_hat, _, results_dict = model(x + self.xi * d, instance_eval=False)
                pred_hat = results_dict[task_name + '_logits']

                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            # pred_hat = model(x + r_adv)
            # logits, results_dict = model(x + r_adv)
            logits, Y_prob, Y_hat, _, results_dict = model(x + r_adv, instance_eval=False)
            pred_hat = results_dict[task_name + '_logits']
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")
       
        return lds

class GaussianNoise(nn.Module):
    """add gasussian noise into feature"""

    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros_ = torch.zeros_like(x)
        n = torch.normal(zeros_, std=self.std)
        return x + n


class PiModelLoss(nn.Module):
    def __init__(self, std=0.15):
        super(PiModelLoss, self).__init__()
        self.gn = GaussianNoise(std)

    def forward(self, model, x):
        logits1 = model(x)
        probs1 = F.softmax(logits1, dim=1)
        with torch.no_grad():
            logits2 = model(self.gn(x))
            probs2 = F.softmax(logits2, dim=1)
        loss = F.mse_loss(probs1, probs2, reduction="sum") / x.size(0)
        # return loss, logits1
        return loss


class DynamicWeight(object):
    def __init__(self, lam, K=3, T=1):
        self.num_loss = 3
        self.loss_t1 = [None, None, None]
        self.loss_t2 = [None, None, None]
        self.w = [None, None, None]
        self.e = [None, None, None]

        self.lam = lam

        self.K, self.T = K, T
        for w in self.lam:
            if w == 0:
                self.K -= 1

    def calc_ratio(self):
        for k in range(self.num_loss):
            if self.lam[k] != 0:
                self.w[k] = self.loss_t1[k] / self.loss_t2[k]
                self.e[k] = math.e ** (self.w[k] / self.T)
            else:
                self.e[k] = 0

        for k in range(self.num_loss):
            self.lam[k] = self.K * self.e[k] / sum(self.e)

    def __call__(self, loss_nega, loss_posi, loss_MIL):
        loss = [loss_nega, loss_posi, loss_MIL]
        for k in range(self.num_loss):
            self.loss_t2[k] = self.loss_t1[k]
            self.loss_t1[k] = loss[k]

        # t = 3, ...
        if self.loss_t2[0] is not None:
            self.calc_ratio()

        return self.lam


def consistency_loss_function(
    consistency_criterion, model, train_loader, img, epoch, task_name
):
   
    consistency_loss = consistency_criterion(model, img, task_name)
    consistency_rampup = 0.4 * epoch * len(train_loader) / 1 # devide by batch size=1
    alpha = get_rampup_weight(0.05, epoch, consistency_rampup)
    consistency_loss = alpha * consistency_loss
    
    return consistency_loss




class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, confidence_threshold=0.9,
                 min_samples_per_class=5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_class = min_samples_per_class

    def forward(self, features, logits):
        device = features.device
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Get pseudo labels and confidence scores
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)
            
            # Get unique classes and their counts
            unique_classes = torch.unique(pseudo_labels)
            selected_indices = []
            
            # Select samples for each class
            for cls in unique_classes:
                cls_indices = torch.where(pseudo_labels == cls)[0]
                cls_confidence = confidence[cls_indices]
                
                # Select top k samples for this class
                k = max(min(int(len(cls_indices) * 0.1), 5),  # Take up to 10% of samples
                       min(self.min_samples_per_class, len(cls_indices)))  # But at least min_samples
                
                if len(cls_confidence) > 0:
                    _, top_k_indices = torch.topk(cls_confidence, k=min(k, len(cls_confidence)))
                    selected_indices.extend(cls_indices[top_k_indices].tolist())
            
            indices = torch.tensor(selected_indices, device=device)
            
            # # Print number of samples per class
            # selected_labels = pseudo_labels[indices]
            # unique_labels, label_counts = torch.unique(selected_labels, return_counts=True)
            # print("\nSelected samples per class:")
            # for label, count in zip(unique_labels.cpu().numpy(), label_counts.cpu().numpy()):
            #     print(f"Class {label}: {count} samples")
        
        # Only use selected samples
        features = features[indices]
        pseudo_labels = pseudo_labels[indices]
        
        batch_size = features.shape[0]
        if batch_size < 2:
            print("Warning: Batch size < 2, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T)
        
        # Temperature scaling with adaptive temperature
        n_views = 2  # Assuming 2 views per sample
        temperature = self.temperature * (batch_size * n_views - 1)  # Scale temperature with batch size
        sim_matrix = sim_matrix / temperature
        
        # Create positive pair mask
        labels = pseudo_labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Count positive pairs
        pos_per_anchor = mask.sum(dim=1)
        
        # # Print positive pairs information
        # total_pairs = pos_per_anchor.sum().item()
        # avg_pairs = total_pairs / batch_size
        # print(f"Total positive pairs: {total_pairs}")
        # print(f"Average positive pairs per anchor: {avg_pairs:.2f}")
        # print(f"Selected batch size: {batch_size}")
        
        # Skip anchors with no positive pairs
        valid_mask = pos_per_anchor > 0
        if not valid_mask.any():
            print("Warning: No valid anchors found (no positive pairs)")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # # Print number of valid anchors
        # num_valid_anchors = valid_mask.sum().item()
        # print(f"Number of valid anchors: {num_valid_anchors}")
        
        # Compute log probabilities with numerical stability
        max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
        sim_matrix = sim_matrix - max_sim
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood for valid anchors only
        mean_log_prob_pos = (mask * log_prob)[valid_mask].sum(dim=1) / (pos_per_anchor[valid_mask] + 1e-8)
        
        # Final loss with scaling
        loss = -mean_log_prob_pos.mean()
        # loss = mean_log_prob_pos.mean()

        return loss