"""
In-memory exact flat dense vector index using PyTorch matrix multiplication.
Guarantees 100% exact retrieval without ANN quantization or pruning distortion.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class DenseIndex:
    """
    In-memory exact flat dense vector index using PyTorch matrix multiplication.
    Supports both CPU and GPU execution.
    """
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.dim = embedding_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.doc_ids: List[str] = []
        self.doc2idx: Dict[str, int] = {}
        self.embeddings: Optional[torch.Tensor] = None

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def build(self, doc_ids: List[str], embeddings: torch.Tensor) -> None:
        """
        Populates index with document IDs and their normalized embedding vectors.

        Args:
            doc_ids: List of N string document identifiers.
            embeddings: Tensor of shape (N, dim) containing document representations.
        """
        if len(doc_ids) != embeddings.size(0):
            raise ValueError(
                f"Mismatch between doc_ids count ({len(doc_ids)}) and embeddings count ({embeddings.size(0)})"
            )

        if self.dim is None:
            self.dim = embeddings.size(1)
        elif embeddings.size(1) != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {embeddings.size(1)}")

        self.doc_ids = list(doc_ids)
        self.doc2idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        emb_device = embeddings.to(self.device).float()
        # Ensure L2 normalized
        norms = torch.norm(emb_device, p=2, dim=-1, keepdim=True)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            emb_device = F.normalize(emb_device, p=2, dim=-1)

        self.embeddings = emb_device

    def search(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 10,
        batch_size: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Executes exact inner product search against index.

        Args:
            query_embeddings: Query tensor of shape (B, dim) or (dim,).
            top_k: Number of top candidate documents to return.
            batch_size: Optional batch size chunking for large query batches.

        Returns:
            List of length B, where each element is a list of [(doc_id, score), ...] tuples
            sorted descending by score.
        """
        if self.embeddings is None or len(self.doc_ids) == 0:
            raise RuntimeError("Cannot search an unbuilt or empty index. Call build() first.")

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        if self.dim is not None and query_embeddings.size(1) != self.dim:
            raise ValueError(f"Query dim {query_embeddings.size(1)} != index dim {self.dim}")

        q_emb = query_embeddings.to(self.device).float()
        # Ensure query embeddings are L2 normalized
        q_norms = torch.norm(q_emb, p=2, dim=-1, keepdim=True)
        if not torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-3):
            q_emb = F.normalize(q_emb, p=2, dim=-1)

        total_queries = q_emb.size(0)
        k = min(top_k, self.num_docs)
        if k <= 0:
            return [[] for _ in range(total_queries)]

        chunk_size = batch_size or total_queries
        all_results: List[List[Tuple[str, float]]] = []

        for start_idx in range(0, total_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, total_queries)
            q_chunk = q_emb[start_idx:end_idx]

            # Exact dot product: (B_chunk, N)
            scores = torch.mm(q_chunk, self.embeddings.t())

            topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
            topk_scores_cpu = topk_scores.cpu().tolist()
            topk_indices_cpu = topk_indices.cpu().tolist()

            for b_idx in range(len(topk_indices_cpu)):
                res = [
                    (self.doc_ids[doc_idx], float(score))
                    for doc_idx, score in zip(topk_indices_cpu[b_idx], topk_scores_cpu[b_idx])
                ]
                all_results.append(res)

        return all_results

    def save(self, file_path: str) -> None:
        """Serializes the index and metadata to disk."""
        torch.save(
            {
                "dim": self.dim,
                "doc_ids": self.doc_ids,
                "embeddings": self.embeddings.cpu() if self.embeddings is not None else None,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str, device: Optional[Union[str, torch.device]] = None) -> "DenseIndex":
        """Loads a serialized index from disk."""
        data = torch.load(file_path, map_location="cpu")
        index = cls(embedding_dim=data.get("dim"), device=device)
        if data.get("embeddings") is not None and data.get("doc_ids"):
            index.build(doc_ids=data["doc_ids"], embeddings=data["embeddings"])
        return index
