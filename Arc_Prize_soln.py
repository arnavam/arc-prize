import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import Optional, Tuple, List

class PatchEmbedding(nn.Module):
    """2D Patch Embedding with ViT-style patching"""
    
    def __init__(self, patch_size: int = 2, d_model: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        # For ARC, channels=1 (grid values)
        batch_size, _, height, width = x.shape
        
        # Pad if necessary to make divisible by patch_size
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Project to embedding dimension
        embeddings = self.projection(patches.to(dtype=torch.float32))
        
        return embeddings

class MambaBlock(nn.Module):

    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Convolutional layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            padding_mode='zeros'
        )
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state).to(torch.float32))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_proj = nn.Linear(self.d_inner, 1, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        
        # Activation functions
        self.act = nn.SiLU()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        # Orthogonal initialization for A
        nn.init.orthogonal_(self.A)
        # Small initialization for dt_proj
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.dt_proj.bias, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Save residual
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # Project to inner dimension
        x_proj = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_conv, x_ssm = x_proj.chunk(2, dim=-1)
        
        # Conv activation and processing
        x_conv = x_conv.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal convolution
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = self.act(x_conv)
        
        # SSM branch
        # Compute parameters
        A = -torch.exp(self.A.to(torch.float32))  # Ensure stability
        D = self.D.to(torch.float32)
        dt = self.dt_proj(x_conv)  # (batch, seq_len, 1)
        dt = nn.functional.softplus(dt)  # Ensure positive dt
        
        B = self.B_proj(x_conv)  # (batch, seq_len, d_state)
        C = self.C_proj(x_conv)  # (batch, seq_len, d_state)
        
        # Discretize A and B
        dA = torch.exp(torch.einsum('bld,ij->blijd', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        
        # Scan through sequence
        action_outputs = []
        for i in range(seq_len):
            h = dA[:, i] @ h + dB[:, i].unsqueeze(-1)
            y_i = torch.einsum('bdn,bn->bd', h, C[:, i]) + D * x_ssm[:, i]
            action_outputs.append(y_i.unsqueeze(1))
        
        x_ssm = torch.cat(action_outputs, dim=1)
        x_ssm = self.act(x_ssm)
        
        # Combine branches
        x = x_conv * x_ssm
        
        # Project back to model dimension
        x = self.out_proj(x)
        
        # Add residual
        x = x + residual
        
        return x

class MambaSSM(nn.Module):
    
    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, n_layers: int = 8, n_classes: int = 10,
                 max_seq_len: int = 1024, patch_size: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        
        # Patch embedding for 2D grids
        self.patch_embed = PatchEmbedding(patch_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Create Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Output layers for classification and position prediction
        self.norm = nn.LayerNorm(d_model)
        self.action_classifier = nn.Linear(d_model, n_classes)
        self.position_predictor = nn.Linear(d_model, 2)  # x, y coordinates
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, current_grid: torch.Tensor, obj_grid: torch.Tensor, target_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shapes: (batch, 1, height, width) - add channel dimension
        batch_size = current_grid.shape[0]
        
        # Process each grid through patch embedding
        current_emb = self.patch_embed(current_grid.unsqueeze(1))  # Add channel dim
        obj_emb = self.patch_embed(obj_grid.unsqueeze(1))
        target_emb = self.patch_embed(target_grid.unsqueeze(1))
        
        # Concatenate embeddings along sequence dimension
        x = torch.cat([current_emb, obj_emb, target_emb], dim=1)
        
        # Add positional encoding
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            # Truncate if necessary
            x = x[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
        
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Process through Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Use the last token's representation for prediction
        x = self.norm(x)
        x = x[:, -1, :]  # Take the last token representation
        
        # Dual outputs
        action_output = self.action_classifier(x)
        position_output = self.position_predictor(x)
        
        return action_output, position_output


from dataset_generator import dataset_creater, create_data_loader
from helper_arc import loader


def train_mamba_model(train_dataset):

    # Hyperparameters
    d_model = 512
    d_state = 16
    d_conv = 4
    expand = 2
    n_layers = 8
    n_classes = 10  # Adjust based on ARC task
    max_seq_len = 1024
    patch_size = 2
    batch_size = 10
    learning_rate = 1e-3
    weight_decay = 0.01
    num_epochs = 50
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MambaSSM(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        n_classes=n_classes,
        max_seq_len=max_seq_len,
        patch_size=patch_size
    ).to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    
    train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)


    # Loss functions
    action_criterion = nn.CrossEntropyLoss()
    pos_criterion = nn.MSELoss()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in pbar:
            print(batch,len(batch))
            current_grids, obj_grids, target_grids, pos_labels , action_labels,  = batch
            
            # Move to device
            current_grids = torch.tensor(current_grids).to(device)
            obj_grids = torch.tensor(obj_grids).to(device)
            target_grids = torch.tensor(target_grids).to(device)
            action_labels = torch.tensor(action_labels).to(device)
            pos_labels = torch.tensor(pos_labels).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            action_outputs, pos_outputs = model(current_grids, obj_grids, target_grids)
            
            # Calculate losses
            action_loss = action_criterion(action_outputs, action_labels)
            pos_loss = pos_criterion(pos_outputs, pos_labels)
            total_loss = action_loss + pos_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_loss += total_loss.item()
            _, predicted = action_outputs.max(1)
            train_total += action_labels.size(0)
            train_correct += predicted.eq(action_labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'action_loss': f'{action_loss.item():.4f}',
                'pos_loss': f'{pos_loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase (you'll need to implement this)
        model.eval()
        scheduler.step()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Accuracy: {100.*train_correct/train_total:.2f}%')
        
        # Add validation code here
        # ...
        
        print('-' * 50)
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    dataset = dataset_creater(create=False) # dataset_creater -> function which creates the dataset.
    train_mamba_model(dataset)

    # train, ids = loader(dataset_path='arc-prize-2025/arc-agi_training_challenges.json')
    # for id in ids:
    #     train_mamba_model(train['id'])