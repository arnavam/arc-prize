import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import  Tuple
from dl_models.mamba import MambaBlock ,ModelArgs
from helper_arc import get_module_logger , plot_metrics
PREDICTION_DICT={}

logger = get_module_logger(__name__)
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
        self.sep_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embedding = nn.Embedding(max_seq_len, d_model) # Positional encoding (learnable)
        self.segment_embedding = nn.Embedding(3, d_model)

        # Create Mamba blocks
        args= ModelArgs(d_model=d_model, d_state=d_state ,d_conv= d_conv,expand= expand)
        self.blocks = nn.ModuleList([
            MambaBlock(args)
            for _ in range(n_layers)
        ])
        
        # Output layers for classification and position prediction
        self.norm = nn.LayerNorm(d_model)
        self.action_classifier = nn.Linear(d_model, n_classes)

        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid()  # Outputs between 0-1, scale to grid size
        )
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
        batch_size=current_grid.shape[0]

        # print('        print(current_grid.shape); ',current_grid.shape)
        current_emb = self.patch_embed(current_grid.unsqueeze(1))  #   Process each grid through patch embedding & Add channel dim
        obj_emb = self.patch_embed(obj_grid.unsqueeze(1))
        target_emb = self.patch_embed(target_grid.unsqueeze(1))
        # print('       print(current_emb.shape)',current_emb.shape)
        # Concatenate embeddings along sequence dimension

        sep_token = self.sep_token_embedding.expand(batch_size, -1, -1) # (batch_size, 1, d_model)
        x = torch.cat([current_emb, sep_token, obj_emb, sep_token, target_emb], dim=1)

        # print('        print(x.shape)   ',x.shape)     


        segment_ids = torch.cat([
            torch.full((current_emb.size(1),), 0),
            torch.full((1,), 1),
            torch.full((obj_emb.size(1),), 1),
            torch.full((1,), 2),
            torch.full((target_emb.size(1),), 2),
        ], dim=0).to(x.device)

        segment_embeddings = self.segment_embedding(segment_ids)  # (seq_len, embed_dim)
        segment_embeddings = segment_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, embed_dim)


        # Add positional encoding
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            # Truncate if necessary
            x = x[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
        pos_embedding = get_sinusoidal_pos_embedding(x.size(1), self.d_model).to(x.device)
        x = x + segment_embeddings 
        x =  x +pos_embedding
        
        # x = x +  + self.pos_embedding[:, :seq_len, :]
        
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



def get_sinusoidal_pos_embedding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0) 


def normalize_grid(grid_data):
    if isinstance(grid_data, list):
        grid_data = np.array(grid_data)  # Convert list to numpy array first
    grid_tensor = torch.tensor(grid_data, dtype=torch.float32)
    return grid_tensor / 10.0  


from dataset_generator import dataset_creater, create_data_loader
from helper_arc import loader


def train_mamba_model(train_dataset,save,load):
    train_losses = []
    train_accuracies = []       
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
    num_epochs = 10
    
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

    if load:
        model.load_state_dict(torch.load('mamba_ssm_model.pth'))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    
    no_of_batch = len(list(train_dataset))


    # Loss functions
    action_criterion = nn.CrossEntropyLoss()
    pos_criterion = torch.nn.SmoothL1Loss()# nn.MSELoss()

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
        train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in pbar:
            current_grids, obj_grids, target_grids, pos_labels , action_labels,  = batch

            
            # Move to device
            current_grids = torch.tensor(current_grids).to(device)
            obj_grids = torch.tensor(obj_grids).to(device)
            target_grids = torch.tensor(target_grids).to(device)
            action_labels = torch.tensor(action_labels).to(device)
            pos_labels = torch.tensor(pos_labels).to(device)

            current_grids = normalize_grid(torch.tensor(current_grids)).to(device)
            obj_grids = normalize_grid(torch.tensor(obj_grids)).to(device)
            target_grids = normalize_grid(torch.tensor(target_grids)).to(device)
            # Forward pass
            optimizer.zero_grad()
            action_outputs, pos_outputs = model(current_grids, obj_grids, target_grids)
            
            # Calculate losses
            action_loss = action_criterion(action_outputs.float(), action_labels)

            pos_loss = pos_criterion(pos_outputs.float(), pos_labels.float())
            total_loss = action_loss + pos_loss

            # Backward pass
            total_loss.backward()
            # After backward pass, check gradients

        # Add gradient checking
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradients in {name}")

            print(f"Gradient norm: {total_grad_norm}")

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
        
            # Log action predictions vs real
            logger.debug(f'[Batch {epoch+1}] Predicted Actions: {predicted.cpu().tolist()}')
            logger.debug(f'[Batch {epoch+1}] Actual Actions:    {action_labels.cpu().tolist()}')

            # Log position predictions vs real
            logger.debug(f'[Batch {epoch+1}] Predicted Positions: {pos_outputs.detach().cpu().numpy().tolist()}')
            logger.debug(f'[Batch {epoch+1}] Actual Positions:    {pos_labels.cpu().tolist()}')


        
        # Validation phase (you'll need to implement this)
        model.eval()
        scheduler.step()
        
        # Print epoch results

        epoch_loss = train_loss / int(no_of_batch*(len(batch)))
        epoch_acc = 100. * train_correct / train_total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Train Accuracy: {epoch_acc:.2f}%')
        print('-' * 50)
        
    if save:
        torch.save(model.state_dict(), 'mamba_ssm_model.pth')
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    plot_metrics(train_losses, train_accuracies)


if __name__ == '__main__':
    dataset = dataset_creater(create=False) # dataset_creater -> function which creates the dataset.
    train_mamba_model(dataset,save=True, load=False)

    # train, ids = loader(dataset_path='arc-prize-2025/arc-agi_training_challenges.json')
    # for id in ids:
    #     train_mamba_model(train['id'])