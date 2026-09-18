
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
from tqdm import tqdm
import os

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Hyperparameters (consistent across phases)
# --------------------------
batch_size = 128
num_workers = 2
epochs_standard = 200
epochs_at = 120
epochs_finetune = 30
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
eps = 8/255
alpha = 2/255
steps = 20
prune_amount = 0.5

# --------------------------
# Data loaders
# --------------------------
def get_cifar10_loaders():
    """Return trainloader and testloader for CIFAR-10 with standard augmentations."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader

# --------------------------
# Training and evaluation helpers
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch, atk=None):
    """Train one epoch, optionally with adversarial training."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        if atk is not None:
            model.eval()
            inputs = atk(inputs, targets)
            model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss': train_loss/total, 'acc': 100.*correct/total})
    return train_loss/total, 100.*correct/total

def evaluate(model, loader, atk=None):
    """Evaluate clean or robust accuracy (single model)."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if atk is not None:
            inputs.requires_grad = True
            adv_inputs = atk(inputs, targets)
            inputs = adv_inputs.detach()
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def evaluate_ensemble(models, loader, atk=None):
    """Ensemble evaluation: average logits of multiple models."""
    for m in models:
        m.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if atk is not None:
            inputs.requires_grad = True
            adv_inputs = atk(inputs, targets)
            inputs = adv_inputs.detach()
        with torch.no_grad():
            outputs = torch.zeros(inputs.size(0), 10).to(device)
            for m in models:
                outputs += m(inputs)
            outputs /= len(models)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def save_checkpoint(model, filename):
    """Save model state dict."""
    torch.save(model.state_dict(), filename)

def load_checkpoint(model, filename):
    """Load model state dict, automatically merging PyTorch prune hooks if present."""
    import os
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")
        
    checkpoint = torch.load(filename)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        base_key = key.replace('_orig', '') if key.endswith('_orig') else key.replace('_mask', '')
        
        # If this param was pruned, we'll reconstruct the physical 'weight' exactly as zeroed
        if key.endswith('_orig'):
            mask_key = base_key + '_mask'
            if mask_key in state_dict:
                reconstructed_weight = value * state_dict[mask_key]
                cleaned_state_dict[base_key] = reconstructed_weight
            else:
                cleaned_state_dict[key] = value
        elif key.endswith('_mask'):
            # Handled dynamically by '_orig' above
            continue
        else:
            cleaned_state_dict[key] = value
            
    model.load_state_dict(cleaned_state_dict)
    return model

def freeze_zeros(model):
    """Re-apply PyTorch prune hooks to preserve zero-valued weights during fine-tuning.
    
    After load_checkpoint strips prune hooks, this function scans all Conv2d layers
    and re-registers prune masks based on which weights are currently zero.
    This ensures SGD optimizer updates cannot overwrite pruned (zero) weights.
    
    Usage: Call immediately after load_checkpoint() and BEFORE creating the optimizer.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip first conv and downsampling layers (not pruned by ExPSO)
            if name == 'conv1' or 'downsample' in name:
                continue
            # Check if this layer has any zeros (i.e., was pruned)
            weight = module.weight.data
            n_zeros = (weight == 0).sum().item()
            if n_zeros > 0:
                # Create binary mask: 1 where weight is non-zero, 0 where pruned
                mask = (weight != 0).float()
                # Re-apply PyTorch's native prune hook to enforce mask during training
                prune.custom_from_mask(module, name='weight', mask=mask)
    
    # Report
    total_pruned = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_mask'):
            total_params += module.weight_mask.numel()
            total_pruned += (module.weight_mask == 0).sum().item()
    if total_params > 0:
        print(f"  freeze_zeros: Locked {total_pruned:,}/{total_params:,} "
              f"({100*total_pruned/total_params:.1f}%) weights at zero")
    return model

def count_zero_params(model):
    """Return percentage of zero-valued parameters in prunable layers (Conv2d + Linear)."""
    total = 0
    zero = 0
    for module in model.modules():
        # PyTorch active prune mask buffer detection
        if hasattr(module, 'weight_mask') and module.weight_mask is not None:
            total += module.weight_mask.numel()
            zero += (module.weight_mask == 0).sum().item()
        elif hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total += module.weight.numel()
                zero += (module.weight == 0).sum().item()
    return zero / total * 100 if total > 0 else 0

def count_conv_sparsity(model):
    """Return sparsity of Conv2d layers exactly accounting for PyTorch prune hooks."""
    total = 0
    zero = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'weight_mask') and module.weight_mask is not None:
                total += module.weight_mask.numel()
                zero += (module.weight_mask == 0).sum().item()
            elif hasattr(module, 'weight') and module.weight is not None:
                total += module.weight.numel()
                zero += (module.weight == 0).sum().item()
    return zero / total * 100 if total > 0 else 0

def print_model_info(model, name="Model"):
    """Print model summary including sparsity."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sparsity = count_zero_params(model)
    
    print(f"\n{name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Non-zero parameters: {int(total_params * (1 - sparsity/100)):,}")