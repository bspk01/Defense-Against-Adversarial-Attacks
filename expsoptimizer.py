"""
ExPSO (Exponential Particle Swarm Optimization) for Neural Network Pruning
Based on: Kraidia et al. (2024) - Defense against adversarial attacks
Adapted for vision models (CNN/ViT)
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
from copy import deepcopy


class ExPSOPruner:
    def __init__(self, model, pruning_ratio=0.5, n_particles=30, max_iter=50,
                 batch_size=32, device='cuda', use_adversarial_fitness=False,
                 epsilon=8/255, alpha=2/255, attack_steps=5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device
        self.use_adversarial_fitness = use_adversarial_fitness
        self.epsilon = epsilon
        self.alpha = alpha
        self.attack_steps = attack_steps

        # ExPSO hyperparameters
        self.a = 2.0
        self.b = 2.0
        self.d = -1.0
        self.e = -1.0
        self.c1 = -1.0
        self.c2 = 2.0
        self.w = 0.9
        self.r = 0.9
        self.k = 0.2

        self.subpop_size = max(1, n_particles // 3)
        self._prunable_layers = []
        self._filter_counts = []

    def _identify_prunable_layers(self):
        self._prunable_layers = []
        self._filter_counts = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name == 'conv1' or 'downsample' in name:
                    continue
                self._prunable_layers.append((name, module))
                self._filter_counts.append(module.out_channels)

        return len(self._prunable_layers)

    def _project_to_binary_mask(self, position, layer_idx):
        scores = np.atleast_1d(position).flatten()
        n_filters = self._filter_counts[layer_idx]
        local_ratio = float(self.pruning_ratio) # Force local float to avoid global clash
        n_keep = max(1, int(n_filters * (1.0 - local_ratio)))

        mask = np.zeros(n_filters, dtype=np.float32)
        if len(scores) >= n_keep:
            top_k_idx = np.argsort(-scores)[:n_keep]
            mask[top_k_idx] = 1.0
        else:
            mask[:] = 1.0

        return mask

    def _fitness(self, masks, data_batches, prev_loss=0.0):
        self.model = self.model.float()

        original_weights = {}
        original_biases = {}
        for name, module in self._prunable_layers:
            original_weights[name] = module.weight.data.clone()
            if module.bias is not None:
                original_biases[name] = module.bias.data.clone()

        for idx, (name, module) in enumerate(self._prunable_layers):
            if idx < len(masks):
                mask = torch.tensor(masks[idx], dtype=torch.float32, device=self.device)
                if isinstance(module, nn.Conv2d):
                    mask = mask.view(-1, 1, 1, 1)
                else:
                    mask = mask.view(-1, 1)
                module.weight.data = module.weight.data * mask
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask.squeeze()

        total_loss = prev_loss
        total_acc = 0.0
        n_samples = 0

        # Implement batch-cumulative performance evaluation
        if not isinstance(data_batches, list):
            data_batches = [data_batches]

        with torch.no_grad():
            self.model.eval()
            for inputs, targets in data_batches:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                batch_loss = nn.CrossEntropyLoss()(outputs, targets).item()
                total_loss += batch_loss
                
                _, pred = outputs.max(1)
                total_acc += pred.eq(targets).float().sum().item()
                n_samples += targets.size(0)

        acc = total_acc / n_samples if n_samples > 0 else 0.0

        total_filters = sum(self._filter_counts)
        kept_filters = sum(np.sum(m) for m in masks)
        compression_ratio = kept_filters / total_filters if total_filters > 0 else 1.0

        if self.use_adversarial_fitness:
            fitness = total_loss - 0.5 * acc + self.pruning_ratio * (1 - compression_ratio)
        else:
            fitness = total_loss + self.pruning_ratio * (1 - compression_ratio) ** 2

        for name, module in self._prunable_layers:
            module.weight.data = original_weights[name]
            if module.bias is not None:
                module.bias.data = original_biases[name]

        return fitness

    def _velocity_damping(self, velocity, iteration):
        """Algorithm 2: Velocity Controller with sigmoid damping and inertia decay."""
        if iteration >= self.max_iter * 0.3:
            vel_max = self.k * self.max_iter
            for i in range(len(velocity)):
                velocity[i] = np.clip(velocity[i], -vel_max, vel_max)
                # Sigmoid-based nonlinear damping (Algorithm 2)
                # Larger velocities are damped more aggressively
                sig = 1.0 / (1.0 + np.exp(-np.clip(np.abs(velocity[i]), -10, 10)))
                velocity[i] = velocity[i] * (1.0 - sig)
            self.w = self.r * self.w
        return velocity

    def _flatten_position(self, position):
        return np.concatenate([np.atleast_1d(p).flatten() for p in position])

    def _update_particle(self, position, velocity, pbest, pworst, gbest, gworst, subpop_idx):
        r1, r2, r3, r4 = np.random.rand(4)

        new_vel = []
        new_pos = []

        for i in range(len(position)):
            pos_i = np.atleast_1d(position[i])
            vel_i = np.atleast_1d(velocity[i])
            pbest_i = np.atleast_1d(pbest[i])
            pworst_i = np.atleast_1d(pworst[i])
            gbest_i = np.atleast_1d(gbest[i])
            gworst_i = np.atleast_1d(gworst[i])

            if subpop_idx == 0:
                # N1: Full exploration (Algorithm 1)
                # Per-layer exponential factor: a * (1 / (||x_i|| + eps))
                norm_i = np.linalg.norm(pos_i)
                exp_factor = self.a * (1.0 / (norm_i + 1e-8))
                v = (self.w * vel_i +
                     exp_factor * r1 * (pbest_i - pos_i) +
                     self.b * r2 * (gbest_i - pos_i) +
                     self.d * r3 * (pworst_i - pos_i) +
                     self.e * r4 * (gworst_i - pos_i))
            elif subpop_idx == 1:
                # N2: Balanced exploration/exploitation
                v = (self.w * vel_i +
                     self.a * r1 * (pbest_i - pos_i) +
                     self.b * r2 * (gbest_i - pos_i))
            else:
                # N3: Exploitation with c1, c2 scaling
                v = (self.w * vel_i +
                     self.c1 * r1 * (pbest_i - pos_i) +
                     self.c2 * r2 * (gbest_i - pos_i))

            p = np.clip(pos_i + v, 0, 1)
            new_vel.append(v.astype(np.float32))
            new_pos.append(p.astype(np.float32))

        return new_pos, new_vel

    def prune(self, model, trainloader, sample_batches=None):
        model = model.float()
        self.model = model

        print("Identifying prunable layers...")
        n_layers = self._identify_prunable_layers()
        if n_layers == 0:
            print("No prunable layers found!")
            return model

        print(f"Found {n_layers} prunable layers (Conv2d + Linear)")

        if sample_batches is None:
            # Gather exactly 3 batches to act as accumulated dataset for fitness
            sample_batches = []
            for b in trainloader:
                inputs, targets = b
                sample_batches.append((inputs[:self.batch_size].float(), targets[:self.batch_size]))
                if len(sample_batches) == 3:
                     break
                     
        importance_scores = []
        for name, module in self._prunable_layers:
            if isinstance(module, nn.Conv2d):
                scores = torch.norm(module.weight.data.float(), p=1, dim=(1, 2, 3)).cpu().numpy()
            else:
                scores = torch.norm(module.weight.data.float(), p=1, dim=1).cpu().numpy()
            importance_scores.append(scores)

        print(f"Initializing swarm with {self.n_particles} particles...")
        positions = []
        velocities = []
        pbest = []
        pbest_fitness = []
        pworst = []
        pworst_fitness = []

        for i in range(self.n_particles):
            particle_pos = []
            particle_vel = []

            for layer_idx, scores in enumerate(importance_scores):
                n_filters = len(scores)
                if i == 0:
                    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                    particle_pos.append(norm_scores.astype(np.float32))
                else:
                    particle_pos.append(np.random.rand(n_filters).astype(np.float32))

                particle_vel.append(np.zeros(n_filters, dtype=np.float32))

            binary_masks = [self._project_to_binary_mask(pos, idx)
                            for idx, pos in enumerate(particle_pos)]
            
            # Fitness calculates batch-cumulative updating loss 
            fitness = self._fitness(binary_masks, sample_batches)

            positions.append(particle_pos)
            velocities.append(particle_vel)
            pbest.append(deepcopy(particle_pos))
            pbest_fitness.append(fitness)
            pworst.append(deepcopy(particle_pos))
            pworst_fitness.append(fitness)

        gbest_idx = np.argmin(pbest_fitness)
        gworst_idx = np.argmax(pworst_fitness)
        gbest = deepcopy(pbest[gbest_idx])
        gbest_fitness = pbest_fitness[gbest_idx]
        gworst = deepcopy(pworst[gworst_idx])
        gworst_fitness = pworst_fitness[gworst_idx]
        
        target_pct = self.pruning_ratio * 100
        print(f"Initial best fitness: {gbest_fitness:.4f}")
        print(f"Running ExPSO swarm optimization for TARGET: {target_pct:.1f}% Sparsity...")
        
        for iteration in tqdm(range(self.max_iter)):
            for i in range(self.n_particles):
                subpop_idx = i // self.subpop_size if self.subpop_size > 0 else 0

                new_pos, new_vel = self._update_particle(
                    positions[i], velocities[i],
                    pbest[i], pworst[i],
                    gbest, gworst, subpop_idx
                )
                new_vel = self._velocity_damping(new_vel, iteration)

                binary_masks = [self._project_to_binary_mask(pos, idx)
                                for idx, pos in enumerate(new_pos)]
                
                # Batch-cumulative fitness (Eq. 1: loss accumulated across sample batches in _fitness)
                new_fitness = self._fitness(binary_masks, sample_batches)

                if new_fitness < pbest_fitness[i]:
                    pbest[i] = deepcopy(new_pos)
                    pbest_fitness[i] = new_fitness

                if new_fitness < gbest_fitness:
                    gbest = deepcopy(new_pos)
                    gbest_fitness = new_fitness

                if new_fitness > pworst_fitness[i]:
                    pworst[i] = deepcopy(new_pos)
                    pworst_fitness[i] = new_fitness

                if new_fitness > gworst_fitness:
                    gworst = deepcopy(new_pos)
                    gworst_fitness = new_fitness

                positions[i] = new_pos
                velocities[i] = new_vel

            if iteration % 10 == 0:
                print(f"Iter {iteration}: Best fitness = {gbest_fitness:.4f}")

        print(f"Final best fitness: {gbest_fitness:.4f}")
        
        # Apply masks permanently
        final_masks = [self._project_to_binary_mask(pos, idx)
                       for idx, pos in enumerate(gbest)]

        for idx, (name, module) in enumerate(self._prunable_layers):
            mask = torch.tensor(final_masks[idx], dtype=torch.float32, device=self.device)
            if isinstance(module, nn.Conv2d):
                mask = mask.view(-1, 1, 1, 1)
            else:
                mask = mask.view(-1, 1)
            
            # Expand mask to perfectly match weight shape for PyTorch's strict hook assertion
            mask = mask.expand_as(module.weight)
            
            # Use PyTorch's native prune hook to successfully freeze weights during fine-tuning
            prune.custom_from_mask(module, name='weight', mask=mask)
            
            if module.bias is not None:
                 pass

        print("Applying optimal pruning masks...")
        for idx, (name, module) in enumerate(self._prunable_layers):
            mask = torch.tensor(final_masks[idx], dtype=torch.float32, device=self.device)
            if isinstance(module, nn.Conv2d):
                mask = mask.view(-1, 1, 1, 1)
            else:
                mask = mask.view(-1, 1)
            module.weight.data = module.weight.data * mask
            if module.bias is not None:
                module.bias.data = module.bias.data * mask.squeeze()

        sparsity = sum((module.weight.data == 0).sum().item()
                       for _, module in self._prunable_layers)
        total = sum(module.weight.numel() for _, module in self._prunable_layers)
        achieved = 100.0 * sparsity / total
        print(f"--> Pruning Complete. Achieved Structural Sparsity: {achieved:.2f}% (Target: {target_pct:.1f}%)")

        return model


def magnitude_prune(model, pruning_ratio=0.5, device='cuda'):
    """
    L1-norm Structured magnitude pruning for Conv2d and Linear layers.
    Ensure this matches ExPSO (which evaluates at the filter/channel level) for fair comparison.
    """
    model = model.float()
    prunable_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == 'conv1':
                continue
            if 'downsample' in name:
                continue
            prunable_layers.append((name, module))

    print(f"Magnitude pruning: Found {len(prunable_layers)} prunable layers")

    for name, module in prunable_layers:
        weight = module.weight.data

        if isinstance(module, nn.Conv2d):
            n_filters = weight.shape[0]
            n_prune = int(n_filters * pruning_ratio)
            n_keep = max(1, n_filters - n_prune)
            l1_norms = torch.norm(weight, p=1, dim=(1, 2, 3))
            _, keep_indices = torch.topk(l1_norms, n_keep)
            mask = torch.zeros(n_filters, device=device)
            mask[keep_indices] = 1.0
            mask = mask.view(-1, 1, 1, 1).expand_as(module.weight)

        # Freeze structurally pruned weights
        prune.custom_from_mask(module, name='weight', mask=mask)

    total_params = 0
    total_zeros = 0
    conv_zeros = 0
    conv_total = 0
    fc_zeros = 0
    fc_total = 0

    for name, module in prunable_layers:
        n_params = module.weight.numel()
        n_zeros = (module.weight == 0).sum().item()
        total_params += n_params
        total_zeros += n_zeros

        if isinstance(module, nn.Conv2d):
            conv_zeros += n_zeros
            conv_total += n_params
        else:
            fc_zeros += n_zeros
            fc_total += n_params

    if total_params > 0:
        print(f"Achieved sparsity: {100.0 * total_zeros / total_params:.2f}%")
    if conv_total > 0:
        print(f"Conv Sparsity: {100.0 * conv_zeros / conv_total:.2f}%")
    if fc_total > 0:
        print(f"FC Sparsity: {100.0 * fc_zeros / fc_total:.2f}%")
    else:
        print("FC Sparsity: 0.00% (Safely Excluded)")
    if total_params > 0:
        print(f"Total Sparsity: {100.0 * total_zeros / total_params:.2f}%")

    return model


def progressive_prune(model, trainloader, target_ratio, n_steps=3,
                      n_particles=30, max_iter=30, batch_size=32,
                      device='cuda', use_adversarial_fitness=True,
                      finetune_epochs=5, finetune_lr=0.01):
    """
    Algorithm 3: Progressive compression with BC-ExPSO.
    Gradually increases pruning ratio from (target/n_steps) to target_ratio.
    Between steps: brief standard fine-tuning to recover accuracy.
    Pruned weights are frozen via PyTorch prune hooks between steps.

    Args:
        model: Pre-trained model to compress
        trainloader: Training data loader
        target_ratio: Final target pruning ratio (e.g. 0.30, 0.50, 0.70)
        n_steps: Number of progressive compression steps
        n_particles: ExPSO swarm size per step
        max_iter: ExPSO iterations per step
        batch_size: Batch size for ExPSO fitness evaluation
        device: CUDA/CPU device
        use_adversarial_fitness: Whether ExPSO uses adversarial fitness
        finetune_epochs: Epochs of standard fine-tuning between steps
        finetune_lr: Learning rate for inter-step fine-tuning

    Returns:
        Progressively pruned model with prune hooks for final ratio
    """
    step_ratios = np.linspace(target_ratio / n_steps, target_ratio, n_steps)
    print(f"\nProgressive compression: {' -> '.join(f'{r:.0%}' for r in step_ratios)}")

    for step_idx, ratio in enumerate(step_ratios):
        print(f"\n{'='*60}")
        print(f"Progressive Step {step_idx+1}/{n_steps}: Pruning to {ratio:.1%}")
        print(f"{'='*60}")

        # Remove prune hooks from previous step (makes pruning permanent in weights)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # Module was not pruned yet

        # Run ExPSO at current progressive ratio
        pruner = ExPSOPruner(
            model=model,
            pruning_ratio=ratio,
            n_particles=n_particles,
            max_iter=max_iter,
            batch_size=batch_size,
            device=device,
            use_adversarial_fitness=use_adversarial_fitness
        )
        model = pruner.prune(model, trainloader)

        # Brief standard fine-tuning between progressive steps (not the last)
        # Paper Algorithm 3: W_t = W_{t-1} - η * ∇L(W_{t-1}, D)
        if step_idx < n_steps - 1 and finetune_epochs > 0:
            print(f"Inter-step fine-tuning ({finetune_epochs} epochs, lr={finetune_lr})...")
            optimizer = torch.optim.SGD(
                model.parameters(), lr=finetune_lr, momentum=0.9, weight_decay=5e-4
            )
            criterion = nn.CrossEntropyLoss()
            model.train()
            for ep in range(finetune_epochs):
                running_loss = 0
                for inputs, targets in trainloader:
                    inputs = inputs.to(device).float()
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(trainloader)
                print(f"  Step-FT Epoch {ep+1}/{finetune_epochs}: loss={avg_loss:.4f}")
            model.eval()

    return model
