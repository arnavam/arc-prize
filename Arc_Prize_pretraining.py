import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import Tuple, Optional
from datetime import datetime
import torchmetrics 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from dsl import ALL_ACTIONS
from dataset_generator import create_dataset, GridDataset
from dl_models.mamba import MambaBlock, ModelArgs
from helper_arc import get_module_logger, plot_metrics , display

PREDICTION_DICT = {}
logger = get_module_logger(__name__)

action_names=list(ALL_ACTIONS.keys())

class PatchEmbedding(nn.Module):
    """2D Patch Embedding with ViT-style patching"""
    
    def __init__(self, patch_size: int = 2, d_model: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Project to embedding dimension
        embeddings = self.projection(patches.to(dtype=torch.float32))
        
        return embeddings

def get_sinusoidal_pos_embedding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional embeddings"""
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class MambaSSM(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 8,
        n_classes: int = 10,
        max_seq_len: int = 1024,
        patch_size: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        batch_size: int = 10,
        grid_size: int = 10,
        action_loss_weight: float = 2.0,
        pos_loss_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[d_state,d_conv,expand,n_layers,n_classes,patch_size])
        
        
        #----------------Embeddings
        self.patch_embed = PatchEmbedding(patch_size, d_model)
        self.sep_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(3, d_model)

        self.log_var_action = nn.Parameter(torch.tensor(0.0))
        self.log_var_pos = nn.Parameter(torch.tensor(0.0))

        #----------------Mamba blocks
        args = ModelArgs(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.blocks = nn.ModuleList([
            MambaBlock(args)
            for _ in range(n_layers)
        ])
        
        #----------------classification and position prediction
        self.norm = nn.LayerNorm(d_model)
        self.action_classifier = nn.Linear(d_model, n_classes)
        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid()
        )
        
        #------------------Loss functions
        self.action_criterion = nn.CrossEntropyLoss()
        self.pos_criterion = nn.SmoothL1Loss()
        
        #------------------metrices
        self.accuracy_metric = torchmetrics.Accuracy(task='multiclass',num_classes=10)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=10)
        self.r2score =torchmetrics.R2Score()
        self.mae = torchmetrics.MeanAbsoluteError()
        #-----------------Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, current_grid: torch.Tensor, obj_grid: torch.Tensor, target_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = current_grid.shape[0]

        # Process each grid through patch embedding
        current_emb = self.patch_embed(current_grid.unsqueeze(1))
        obj_emb = self.patch_embed(obj_grid.unsqueeze(1))
        target_emb = self.patch_embed(target_grid.unsqueeze(1))

        # Concatenate embeddings along sequence dimension
        sep_token = self.sep_token_embedding.expand(batch_size, -1, -1)
        x = torch.cat([current_emb, sep_token, obj_emb, sep_token, target_emb], dim=1)

        # Segment embeddings
        segment_ids = torch.cat([
            torch.full((current_emb.size(1),), 0),
            torch.full((1,), 1),
            torch.full((obj_emb.size(1),), 1),
            torch.full((1,), 2),
            torch.full((target_emb.size(1),), 2),
        ], dim=0).to(x.device)

        segment_embeddings = self.segment_embedding(segment_ids)
        segment_embeddings = segment_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional encoding
        seq_len = x.shape[1]
        if seq_len > self.hparams.max_seq_len:
            x = x[:, :self.hparams.max_seq_len, :]
            seq_len = self.hparams.max_seq_len
            
        pos_embedding = get_sinusoidal_pos_embedding(x.size(1), self.hparams.d_model).to(x.device)
        x = x + segment_embeddings + pos_embedding
        
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

    def _shared_step(self, batch, batch_idx, train=True):
        current_grids, obj_grids, target_grids, pos_labels, action_labels = batch

        # Forward pass
        action_outputs, pos_outputs = self(current_grids, obj_grids, target_grids)
        
        # Calculate losses
        action_loss = self.action_criterion(action_outputs, action_labels)
        pos_loss = self.pos_criterion(pos_outputs, pos_labels)
        total_loss = self.hparams.action_loss_weight * action_loss + self.hparams.pos_loss_weight * pos_loss
        loss = (torch.exp(-self.log_var_action) * action_loss + self.log_var_action) + \
               (torch.exp(-self.log_var_pos) * pos_loss + self.log_var_pos)

        # Calculate metrics
        self.eval()
        _, predicted = action_outputs.max(1)
        accuracy= self.accuracy_metric(predicted, action_labels)
        f1score = self.f1_metric(predicted, action_labels)
        # display(current_grids[1].detach().cpu().numpy(),obj_grids[1].detach().cpu().numpy(),target_grids[1].detach().cpu().numpy(),predicted_title=pos_outputs[1].detach().cpu().numpy(),target_title=action_names[action_outputs[1].detach().cpu().numpy()],folder='debug',printing=False)

        pos_mae = torch.mean(torch.abs(pos_outputs - pos_labels)).item()
        pos_r2 = self.r2score(pos_outputs,pos_labels)
        # Log metrics
        name = 'train' if train else 'val'
        self.log(f'{name}_loss', total_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(f'{name}_action_loss', action_loss, prog_bar=False, logger=True, on_epoch=True)
        self.log(f'{name}_pos_loss', pos_loss, prog_bar=False, logger=True, on_epoch=True)
        self.log(f'{name}_accuracy', accuracy, prog_bar=True, logger=True, on_epoch=True)        
        self.log(f'{name}_f1loss', f1score, prog_bar=False, logger=True, on_epoch=True)
        self.log(f'{name}_mae', pos_mae, prog_bar=False, logger=True, on_epoch=True)
        self.log(f'{name}_r2score', pos_r2, prog_bar=True, logger=True, on_epoch=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, train=True)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, train=False)



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _log_predictions(self, action_outputs, action_labels, pos_outputs, pos_labels_norm, batch_idx):
        """Log predictions for debugging purposes"""
        # Log action predictions
        _, predicted_actions = action_outputs.max(1)
        logger.debug(f'[Batch {batch_idx}] Predicted Actions: {predicted_actions.cpu().tolist()}')
        logger.debug(f'[Batch {batch_idx}] Actual Actions: {action_labels.cpu().tolist()}')
        
        # Log position predictions
        logger.debug(f'[Batch {batch_idx}] Predicted Positions: {pos_outputs.detach().cpu().numpy().tolist()}')
        logger.debug(f'[Batch {batch_idx}] Actual Positions: {pos_labels_norm.float().cpu().tolist()}')

class GridDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_grids,
        obj_grids,
        target_grids,
        obj_positions,
        action_labels,
        batch_size: int = 10,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        self.hparams.input_grids = np.array(self.hparams.input_grids) / 9.0
        self.hparams.obj_grids = np.array(self.hparams.obj_grids) / 9.0
        self.hparams.target_grids = np.array(self.hparams.target_grids) / 9.0

        grid_size = self.hparams.target_grids.shape[-1] - 1
        self.hparams.obj_positions = np.array(self.hparams.obj_positions) / grid_size

        full_dataset = GridDataset(
            self.hparams.input_grids ,
            self.hparams.obj_grids ,
            self.hparams.target_grids, 
            self.hparams.obj_positions ,
            self.hparams.action_labels,
        )
            
        # Calculate split sizes
        dataset_size = len(full_dataset)
        print(dataset_size)
        train_size = int(self.hparams.train_ratio * dataset_size)
        val_size = int(self.hparams.val_ratio * dataset_size)
        
        # Split dataset
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            pin_memory=True,
            persistent_workers=True,
            num_workers=min(4, os.cpu_count() or 1)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            pin_memory=True,
            persistent_workers=True,
            num_workers=min(4, os.cpu_count() or 1)
        )

import optuna
  


def train_mamba_model(config, train_dataset, save=True, load=False):


    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_tb = TensorBoardLogger(
        save_dir='runs',
        name=f'{timestamp}',
        version=''
    )
    logger_tb.log_hyperparams(config)   
    
    # Create data module
    input_grids, obj_grids, target_grids, obj_positions, action_labels = train_dataset
    data_module = GridDataModule(
        input_grids, obj_grids, target_grids, obj_positions, action_labels,
        batch_size=config['batch_size']
    )
    
    # Create model
    model = MambaSSM(**{k: v for k, v in config.items() if k not in ['max_epochs']})
    
    # Load checkpoint if requested
    if load and os.path.exists('mamba_ssm_model.pth'):
        state_dict = torch.load('mamba_ssm_model.pth')
        model.load_state_dict(state_dict)
        print("Loaded pre-trained weights")
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Setup logging and callbacks
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_tb = TensorBoardLogger(f'runs/{timestamp}', name='mamba_ssm')
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=f'checkpoints/{timestamp}',
            filename='mamba-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'        
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=logger_tb,
        callbacks=callbacks,
        log_every_n_steps=1,
        accelerator='auto',
        enable_progress_bar=save,
        enable_model_summary=save,
        gradient_clip_val=1.0
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    
    # Save the final model
    if save:
        torch.save(model.state_dict(), 'mamba_ssm_model.pth')
        print("Model saved to mamba_ssm_model.pth")
    
    return trainer.callback_metrics["val_loss"].item()

def finetune_mamba_model(train_dataset,save,load):
    def create(trial):
        params = {
            "d_model": trial.suggest_categorical("d_model", [256, 512, 768]),
            "d_state": trial.suggest_int("d_state", 8, 32),
            "d_conv": trial.suggest_int("d_conv", 2, 8),
            "expand": trial.suggest_int("expand", 1, 4),
            "n_layers": trial.suggest_int("n_layers", 1, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "action_loss_weight": trial.suggest_float("action_loss_weight", 1.0, 5.0),
            "pos_loss_weight": trial.suggest_float("pos_loss_weight", 0.5, 3.0),

            # Fixed parameters
            "grid_size": 10,
            "n_classes": 10,
            "max_seq_len": 1024,
            "patch_size": 2,
            "max_epochs":30
            
        }
        return params

    if save:
        with open("config.json", "r") as f:
            best_config = json.load(f)   
            val_loss =train_mamba_model(best_config, train_dataset, save=save, load=load)

    else:

        study = optuna.create_study(direction="minimize")
        study.optimize(
        lambda trial: train_mamba_model(create(trial), train_dataset, save=save, load=load),
        n_trials=50)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        with open("config.json", "w") as f:
            json.dump(study.best_trial, f, indent=4)  


if __name__ == '__main__':
    # Create dataset
    input_grids, obj_grids, target_grids, obj_positions, action_labels = create_dataset(
        create=False,
        num_simple_tasks=20,
        num_intermediate_tasks=80,
        grid_size=(10, 10),
        num_bg_objects=5,
        simple_examples_per_task=5,
        intermediate_examples_per_task=5,
    )
    finetune_mamba_model(
        (input_grids, obj_grids, target_grids, obj_positions, action_labels),
        save=True, 
        load=False)




