import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

def compute_embedding_stability(layer_embeddings: List[torch.Tensor], query_idx: int) -> List[float]:
    similarities = []
    for i in range(1, len(layer_embeddings)):
        emb1 = layer_embeddings[i-1][query_idx].unsqueeze(0)
        emb2 = layer_embeddings[i][query_idx].unsqueeze(0)
        sim = F.cosine_similarity(emb1, emb2).item()
        similarities.append(sim)
    return similarities

def compute_retrieval_margin(scores: List[Tuple[str, float]]) -> float:
    if len(scores) < 2:
        return 0.0
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][1] - sorted_scores[1][1]

def compute_score_concentration(scores: List[Tuple[str, float]], top_k: int = 5) -> float:
    if len(scores) <= top_k:
        return 1.0
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k_mean = sum(s[1] for s in sorted_scores[:top_k]) / top_k
    rest_mean = sum(s[1] for s in sorted_scores[top_k:]) / (len(sorted_scores) - top_k)
    
    if rest_mean == 0:
        return float('inf')
    return top_k_mean / rest_mean

def classify_query_difficulty(stability_scores: List[float], retrieval_margin: float, threshold: float = 0.95) -> str:
    early_stable = False
    for i, sim in enumerate(stability_scores):
        if sim >= threshold:
            if i < len(stability_scores) // 2:
                early_stable = True
            break
            
    if early_stable and retrieval_margin > 0.1:
        return 'easy'
    elif not early_stable and retrieval_margin < 0.05:
        return 'hard'
    else:
        return 'medium'

def analyze_query_characteristics(query_text: str) -> Dict[str, Any]:
    words = query_text.lower().split()
    length = len(words)
    ambiguous_words = {'it', 'that', 'which', 'this', 'they', 'them', 'their'}
    ambiguity_count = sum(1 for w in words if w in ambiguous_words)
    
    return {
        'length': length,
        'num_terms': length,
        'ambiguity_score': ambiguity_count / max(1, length),
        'has_ambiguous_words': ambiguity_count > 0,
        'unique_terms': len(set(words))
    }
