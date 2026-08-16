import random
from typing import List, Dict, Any, Optional

class ExitPolicy:
    def should_exit(self, layer_idx: int, **kwargs) -> bool:
        raise NotImplementedError
        
    def reset(self):
        pass

class StabilityExitPolicy(ExitPolicy):
    def __init__(self, threshold: float, min_layer: int, patience: int):
        self.threshold = threshold
        self.min_layer = min_layer
        self.patience = patience
        self._consecutive_stable = 0

    def should_exit(self, layer_idx: int, similarities: Optional[List[float]] = None, **kwargs) -> bool:
        if similarities and len(similarities) > 0:
            latest_sim = similarities[-1]
            if latest_sim >= self.threshold:
                self._consecutive_stable += 1
            else:
                self._consecutive_stable = 0
        
        if layer_idx >= self.min_layer and self._consecutive_stable >= self.patience:
            return True
        return False
        
    def reset(self):
        self._consecutive_stable = 0

class FixedDepthPolicy(ExitPolicy):
    def __init__(self, exit_layer: int):
        self.exit_layer = exit_layer
        
    def should_exit(self, layer_idx: int, **kwargs) -> bool:
        return layer_idx >= self.exit_layer
        
    def reset(self):
        pass

class RandomExitPolicy(ExitPolicy):
    def __init__(self, min_layer: int, max_layer: int, seed: Optional[int] = None):
        self.min_layer = min_layer
        self.max_layer = max_layer
        if seed is not None:
            random.seed(seed)
        self._exit_layer = random.randint(self.min_layer, self.max_layer)
        
    def should_exit(self, layer_idx: int, **kwargs) -> bool:
        return layer_idx >= self._exit_layer
        
    def reset(self):
        self._exit_layer = random.randint(self.min_layer, self.max_layer)

class OracleExitPolicy(ExitPolicy):
    def __init__(self, optimal_layers: Dict[int, int]):
        self.optimal_layers = optimal_layers
        
    def should_exit(self, layer_idx: int, query_idx: int = -1, **kwargs) -> bool:
        exit_layer = self.optimal_layers.get(query_idx, float('inf'))
        return layer_idx >= exit_layer
        
    def reset(self):
        pass
