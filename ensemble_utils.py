"""
Ensemble utilities for adversarial defense
Creates diverse compressed models for robust ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import torchattacks

def create_diverse_models(base_model, n_models=5, pruning_ratios=[0.3, 0.4, 0.5, 0.6, 0.7],
                         device='cuda', trainloader=None, sample_batch=None):
    models = []
    masks_list = []
    
    importance_scores = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv1' not in name:
            scores = torch.norm(module.weight.data, p=1, dim=(1,2,3)).cpu().numpy()
            importance_scores.append((name, scores))
    
    for i, ratio in enumerate(pruning_ratios[:n_models]):
        model_copy = deepcopy(base_model)
        masks = {}
        
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv1' not in name:
                scores = torch.norm(module.weight.data, p=1, dim=(1,2,3)).cpu().numpy()
                n_filters = len(scores)
                n_keep = max(1, int(n_filters * (1 - ratio)))
                
                random_factor = np.random.rand() * 0.3
                adjusted_scores = scores * (1 + random_factor * (np.random.rand(*scores.shape) - 0.5))
                
                keep_idx = np.argsort(-adjusted_scores)[:n_keep]
                mask = torch.zeros(n_filters, dtype=torch.float32, device=device)
                mask[keep_idx] = 1.0
                
                masks[name] = mask
                module.weight.data = module.weight.data * mask.view(-1, 1, 1, 1)
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask
        
        models.append(model_copy)
        masks_list.append(masks)
        print(f"Model {i+1}: pruning ratio={ratio:.0%}")
    
    return models, masks_list

def ensemble_predict(models, inputs):
    with torch.no_grad():
        outputs = torch.zeros(inputs.size(0), 10).to(inputs.device)
        for model in models:
            model.eval()
            outputs += model(inputs)
        outputs /= len(models)
    return outputs

def evaluate_ensemble(models, loader, atk=None, device='cuda'):
    correct = 0
    total = 0
    for model in models:
        model.eval()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if atk is not None:
            class EnsembleWrapper(nn.Module):
                def __init__(self, models):
                    super().__init__()
                    self.models = models
                def forward(self, x):
                    out = torch.zeros(x.size(0), 10).to(x.device)
                    with torch.no_grad():
                        for m in self.models:
                            out += m(x)
                    return out / len(self.models)
            
            ensemble_for_attack = EnsembleWrapper(models)
            attacker = torchattacks.PGD(ensemble_for_attack, eps=atk['eps'], 
                                        alpha=atk['alpha'], steps=atk['steps'])
            inputs = attacker(inputs, targets)
        
        with torch.no_grad():
            outputs = ensemble_predict(models, inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def load_pretrained_models(checkpoint_paths, model_class, device='cuda'):
    models = []
    for path in checkpoint_paths:
        model = model_class(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)
    return models

class SME_Ensemble(nn.Module):
    """
    Stochastic Multi-Expert (SME) Ensemble — Phase 3.
    K compressed expert models with Gumbel-Softmax gating.
    Paper Eq. 10-11 (full version):
      w = W^T(ỹ ⊕ f(x, θ*)) + b   (gate sees expert OUTPUTS + FEATURES)
      α = Gumbel-Softmax(w/τ)       (stochastic selection)
      ŷ = (1/K) * Σ(αj * ỹj)       (weighted aggregation, Eq. 9)

    For vision: extracts 512-dim features from each expert's avgpool layer
    via forward hooks, giving the gate rich spatial information alongside logits.
    Gate input: K*num_classes (logits) + K*512 (features) = 1566-dim for K=3.
    """
    def __init__(self, experts, num_classes=10, feature_dim=512, train_experts=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.k = len(experts)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Freeze experts by default
        if not train_experts:
            for param in self.experts.parameters():
                param.requires_grad = False

        # Storage for intermediate features captured by hooks
        self._features = [None] * self.k

        # Register forward hooks on each expert's avgpool to capture features
        for idx, expert in enumerate(self.experts):
            self._register_feature_hook(expert, idx)

        # Gating MLP (Eq. 10): w = W^T(ỹ ⊕ f(x, θ*)) + b
        # Input: expert logits (K * num_classes) + expert features (K * feature_dim)
        gate_input_dim = self.k * num_classes + self.k * feature_dim
        gate_hidden_dim = 512
        self.gate_W = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, self.k)
        )

    def _register_feature_hook(self, expert, idx):
        """Register a hook on the expert's avgpool layer to capture features."""
        def hook_fn(module, input, output, expert_idx=idx):
            self._features[expert_idx] = output.flatten(1)  # (batch, 512)

        # ResNet-18: avgpool is the second-to-last layer
        expert.avgpool.register_forward_hook(hook_fn)

    def forward(self, x, tau=1.0, hard=False):
        # 1. Get outputs from all K experts (hooks capture features automatically)
        expert_outputs = [expert(x) for expert in self.experts]
        y_tilde = torch.stack(expert_outputs, dim=1)   # (batch, K, num_classes)
        y_concat = torch.cat(expert_outputs, dim=-1)    # (batch, K * num_classes)

        # 2. Concatenate features from all experts: f(x, θ*) for each expert
        features_concat = torch.cat(self._features, dim=-1)  # (batch, K * 512)

        # 3. Full Eq. 10: w = W^T(ỹ ⊕ f(x, θ*)) + b
        gate_input = torch.cat([y_concat, features_concat], dim=-1)  # (batch, K*M + K*Q)
        w = self.gate_W(gate_input)  # (batch, K)

        # 4. Stochastic selection (Eq. 11): α = Gumbel-Softmax(w / τ)
        alpha = F.gumbel_softmax(w, tau=tau, hard=hard)  # (batch, K)

        # 5. Aggregation (Eq. 9): ŷ = (1/K) * Σ(αj * ỹj)
        alpha = alpha.unsqueeze(-1)  # (batch, K, 1)
        output = (1.0 / self.k) * torch.sum(alpha * y_tilde, dim=1)  # (batch, num_classes)

        return output


class MVC_NNT(nn.Module):
    """
    Multi-Version Compressed Neural Network Training — Phase 4.
    Random model selection at inference for adversarial defense.
    Paper: Fig. 8, Eq. 12.
    At each forward pass, one model is randomly selected, making
    it impossible for an attacker to reliably target a specific model.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.k = len(models)

    def forward(self, x):
        """Randomly select ONE model for prediction (paper's inference strategy)."""
        idx = np.random.randint(0, self.k)
        return self.models[idx](x)

    def evaluate(self, loader, device='cuda', n_trials=5):
        """Average accuracy over n_trials of random-selection inference."""
        self.eval()
        total_correct = 0
        total_samples = 0
        for _ in range(n_trials):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs = self.forward(inputs)
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    total_samples += targets.size(0)
        return 100. * total_correct / total_samples