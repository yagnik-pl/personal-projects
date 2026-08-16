"""
DenseRanker: Coordinates encoding and retrieval operations against the DenseIndex.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from src.data.schemas import CorpusEntry, QueryEntry
from src.retrieval.index import DenseIndex


class DenseRanker:
    """
    High-level orchestrator for encoding text collections and executing dense retrieval runs.
    """
    def __init__(self, encoder: Any, index: Optional[DenseIndex] = None):
        self.encoder = encoder
        self.index = index or DenseIndex(
            embedding_dim=getattr(encoder, "hidden_dim", None),
            device=getattr(encoder, "device", None),
        )

    def index_corpus(
        self,
        corpus: Dict[str, CorpusEntry],
        layer: int = -1,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        """
        Encodes all corpus documents at layer `layer` and builds the dense index.

        Args:
            corpus: Dict mapping doc_id -> CorpusEntry.
            layer: Layer index to extract (-1 for last layer).
            batch_size: Batch size for document encoding.
            show_progress: Whether to show progress bar.
        """
        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id].full_text for doc_id in doc_ids]

        target_layer = self.encoder.num_layers if layer == -1 else layer
        doc_embeddings = self.encoder.encode_layer(
            texts=texts,
            layer=target_layer,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        self.index.build(doc_ids=doc_ids, embeddings=doc_embeddings)

    def rank_queries(
        self,
        queries: Dict[str, QueryEntry],
        top_k: int = 10,
        layer: int = -1,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Encodes all queries at layer `layer` and retrieves top-k documents per query.

        Args:
            queries: Dict mapping query_id -> QueryEntry.
            top_k: Number of candidate documents to retrieve.
            layer: Layer index to extract (-1 for last layer).
            batch_size: Batch size for query encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Dict mapping query_id -> [(doc_id, score), ...]
        """
        qids = list(queries.keys())
        texts = [queries[qid].text for qid in qids]

        target_layer = self.encoder.num_layers if layer == -1 else layer
        q_embeddings = self.encoder.encode_layer(
            texts=texts,
            layer=target_layer,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        search_results = self.index.search(q_embeddings, top_k=top_k)
        return {qid: res for qid, res in zip(qids, search_results)}
