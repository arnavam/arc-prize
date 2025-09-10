import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import Optional, Tuple

class MambaBlock(nn.Module):
    """Single Mamba block with selective state space mechanism"""
    
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
        outputs = []
        for i in range(seq_len):
            h = dA[:, i] @ h + dB[:, i].unsqueeze(-1)
            y_i = torch.einsum('bdn,bn->bd', h, C[:, i]) + D * x_ssm[:, i]
            outputs.append(y_i.unsqueeze(1))
        
        x_ssm = torch.cat(outputs, dim=1)
        x_ssm = self.act(x_ssm)
        
        # Combine branches
        x = x_conv * x_ssm
        
        # Project back to model dimension
        x = self.out_proj(x)
        
        # Add residual
        x = x + residual
        
        return x

class MambaSSM(nn.Module):
    """
    Mamba State Space Model for ARC Prize competition
    """
    
    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, n_layers: int = 8, n_classes: int = 10,
                 vocab_size: Optional[int] = None, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer if dealing with discrete tokens
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Create Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Output layers for classification
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Input embedding if dealing with tokens
        if self.embedding is not None:
            x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Process through Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification head (use the last token's representation)
        x = self.norm(x)
        x = x[:, -1, :]  # Take the last token representation
        x = self.classifier(x)
        
        return x

class ARCDataset(Dataset):
    """Dataset for ARC Prize competition"""
    
    def __init__(self, data_dir, split='train', max_seq_len=1024):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load data
        with open(os.path.join(data_dir, f'{split}.json'), 'r') as f:
            self.data = json.load(f)
        
        # Preprocess data
        self.examples = self._preprocess_data()
    
    def _preprocess_data(self):
        examples = []
        for item in self.data:
            # Convert input to token sequence (this is a placeholder)
            # You'll need to implement the actual tokenization for ARC tasks
            input_seq = self._tokenize_input(item['input'])
            label = item['output']
            
            # Pad or truncate sequence
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[:self.max_seq_len]
            else:
                input_seq = input_seq + [0] * (self.max_seq_len - len(input_seq))
            
            examples.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            })
        
        return examples
    
    def _tokenize_input(self, input_data):
        """
        Tokenize ARC input data. This is a placeholder implementation.
        You'll need to implement the actual tokenization strategy for ARC tasks.
        """
        # For demonstration, we'll use a simple approach
        # In practice, you'll need a more sophisticated tokenization strategy
        if isinstance(input_data, list):
            # Flatten the grid and convert to tokens
            tokens = []
            for row in input_data:
                for cell in row:
                    tokens.append(cell + 1)  # Add 1 to avoid 0 (padding)
            return tokens
        else:
            # Handle other input formats
            return [int(x) for x in str(input_data)]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]['input'], self.examples[idx]['label']

def train_mamba_arc():
    """Train Mamba model on ARC Prize data"""
    # Hyperparameters
    d_model = 512
    d_state = 16
    d_conv = 4
    expand = 2
    n_layers = 8
    n_classes = 10  # Adjust based on ARC task
    vocab_size = 100  # Adjust based on tokenization
    max_seq_len = 256
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50
    data_dir = 'arc_data'  # Path to your ARC data
    
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
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create datasets
    train_dataset = ARCDataset(data_dir, 'train', max_seq_len)
    val_dataset = ARCDataset(data_dir, 'val', max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
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
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mamba_arc.pth')
            print(f'New best model saved with accuracy: {best_val_acc:.2f}%')
        
        print('-' * 50)
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    # Create a sample config for ARC data directory structure
    # You'll need to download the ARC data and organize it appropriately
    sample_config = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": 0},
            {"input": [[5, 6], [7, 8]], "output": 1}
        ],
        "val": [
            {"input": [[9, 10], [11, 12]], "output": 0},
            {"input": [[13, 14], [15, 16]], "output": 1}
        ]
    }
    
    # Create data directory if it doesn't exist
    os.makedirs('arc_data', exist_ok=True)
    
    # Save sample config (replace with actual ARC data)
    with open('arc_data/train.json', 'w') as f:
        json.dump(sample_config['train'], f)
    
    with open('arc_data/val.json', 'w') as f:
        json.dump(sample_config['val'], f)
    
    # Train the model
    train_mamba_arc()