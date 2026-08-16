import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import random

class LearnedExitController(nn.Module):
    def __init__(self, hidden_dim: int, controller_hidden: int = 128, num_features: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_features, controller_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embedding: torch.Tensor, stability: float, layer_idx: int, max_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        batch_size = embedding.size(0)
        device = embedding.device
        
        stability_tensor = torch.full((batch_size, 1), float(stability), device=device)
        norm_layer = float(layer_idx) / max_layers
        layer_tensor = torch.full((batch_size, 1), norm_layer, device=device)
        
        scalar_features = [stability_tensor, layer_tensor]
        while len(scalar_features) < self.num_features:
            scalar_features.append(torch.zeros_like(stability_tensor))
            
        scalars = torch.cat(scalar_features[:self.num_features], dim=1)
        
        x = torch.cat([embedding, scalars], dim=1)
        prob = self.mlp(x).squeeze(-1)
        should_exit = prob > 0.5
        
        return prob, should_exit

class ControllerTrainer:
    def __init__(self, controller: LearnedExitController, learning_rate: float = 0.001, device: str = 'cpu'):
        self.controller = controller.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.controller.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def generate_oracle_labels(self, layer_metrics: List[Dict], full_depth_metrics: Dict, quality_threshold: float = 0.95) -> List[int]:
        oracle_layers = []
        for i, q_metrics in enumerate(layer_metrics):
            full_score = full_depth_metrics.get(i, full_depth_metrics.get(str(i), 0.0))
            target_score = quality_threshold * full_score
            
            best_layer = max([int(k) for k in q_metrics.keys()]) if q_metrics else 12
            for layer_idx in sorted([int(k) for k in q_metrics.keys()]):
                if q_metrics[layer_idx] >= target_score or q_metrics[layer_idx] >= full_score:
                    best_layer = layer_idx
                    break
            oracle_layers.append(best_layer)
        return oracle_layers
        
    def train(self, training_data: List[Dict], epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        self.controller.train()
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                self.optimizer.zero_grad()
                batch_loss_val = 0.0
                loss_tensors = []
                
                for item in batch:
                    embedding = item['embedding'].to(self.device)
                    stability = item['stability']
                    layer_idx = item['layer_idx']
                    max_layers = item.get('max_layers', 12)
                    label = torch.tensor([1.0 if item['should_exit'] else 0.0], device=self.device)
                    
                    prob, _ = self.controller(embedding, stability, layer_idx, max_layers)
                    loss = self.criterion(prob.unsqueeze(0) if prob.dim() == 0 else prob, label)
                    loss_tensors.append(loss)
                    
                batch_loss = torch.stack(loss_tensors).mean()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                
            history['loss'].append(epoch_loss / max(1, len(training_data) // batch_size))
            
        return history
        
    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        self.controller.eval()
        correct = 0
        total = len(eval_data)
        
        with torch.no_grad():
            for item in eval_data:
                embedding = item['embedding'].to(self.device)
                stability = item['stability']
                layer_idx = item['layer_idx']
                max_layers = item.get('max_layers', 12)
                label = item['should_exit']
                
                _, should_exit = self.controller(embedding, stability, layer_idx, max_layers)
                if should_exit.item() == label:
                    correct += 1
                    
        return {'accuracy': correct / max(1, total)}
