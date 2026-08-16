"""
LayerWiseEncoder: Transformer encoder with intermediate hidden state extraction
and step-by-step sequential layer execution for query-adaptive dynamic early exit.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.models.pooling import cls_pooling, mean_pooling, normalize_embeddings


class LayerWiseEncoder(nn.Module):
    """
    Dense retriever encoder wrapper supporting:
    1. Automatic architecture resolution (BERT, MiniLM, MPNet).
    2. Batched all-layer extraction (layers 0..L) for corpus indexing.
    3. Step-by-step sequential layer execution for early-exit query inference.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        pooling_strategy: Optional[str] = None,
        max_length: int = 512,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.max_length = max_length

        # Device resolution
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Automated pooling strategy detection if not explicitly set
        if pooling_strategy is not None:
            self.pooling_strategy = pooling_strategy.lower()
        else:
            lower_name = model_name_or_path.lower()
            if "bge" in lower_name:
                self.pooling_strategy = "cls"
            elif "minilm" in lower_name or "mpnet" in lower_name or "contriever" in lower_name:
                self.pooling_strategy = "mean"
            else:
                self.pooling_strategy = "cls"

        if self.pooling_strategy not in ("cls", "mean"):
            raise ValueError(
                f"Unsupported pooling strategy: '{self.pooling_strategy}'. Must be 'cls' or 'mean'."
            )

        # Load HuggingFace Tokenizer and Base Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        self.num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        self.hidden_dim = getattr(self.model.config, "hidden_size", 384)

    def pool_and_normalize(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pools sequence hidden state and applies L2 normalization.
        """
        if self.pooling_strategy == "cls":
            pooled = cls_pooling(hidden_state)
        elif self.pooling_strategy == "mean":
            pooled = mean_pooling(hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        return normalize_embeddings(pooled, p=2.0, eps=1e-9)

    @torch.no_grad()
    def encode_all_layers(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """
        Extracts L2-normalized embeddings for ALL layers (0..L).

        Args:
            texts: List of N input strings.
            batch_size: Number of texts per forward pass.
            show_progress: Whether to display a progress bar.

        Returns:
            List of (L + 1) CPU tensors, each of shape (N, hidden_dim).
        """
        if not texts:
            return [torch.empty((0, self.hidden_dim), dtype=torch.float32) for _ in range(self.num_layers + 1)]

        all_layer_embs: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers + 1)]

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding all layers")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded, output_hidden_states=True)
            # outputs.hidden_states has length num_layers + 1 (layer 0 = embedding, layers 1..L = transformer outputs)
            for l_idx, h_state in enumerate(outputs.hidden_states):
                norm_emb = self.pool_and_normalize(h_state, encoded["attention_mask"])
                all_layer_embs[l_idx].append(norm_emb.cpu())

        return [torch.cat(layer_batches, dim=0) for layer_batches in all_layer_embs]

    @torch.no_grad()
    def encode_layer(
        self,
        texts: List[str],
        layer: int,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Extracts L2-normalized embeddings at a specific layer index l in {0..L}.

        Args:
            texts: List of N input strings.
            layer: Integer layer index in [0, num_layers].
            batch_size: Batch size for tokenization and model pass.
            show_progress: Whether to display progress bar.

        Returns:
            CPU Tensor of shape (N, hidden_dim).
        """
        if not (0 <= layer <= self.num_layers):
            raise ValueError(f"Layer {layer} out of valid bounds [0, {self.num_layers}]")

        if not texts:
            return torch.empty((0, self.hidden_dim), dtype=torch.float32)

        layer_embs: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Encoding layer {layer}")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded, output_hidden_states=True)
            h_state = outputs.hidden_states[layer]
            norm_emb = self.pool_and_normalize(h_state, encoded["attention_mask"])
            layer_embs.append(norm_emb.cpu())

        return torch.cat(layer_embs, dim=0)

    def _forward_transformer_layer(
        self,
        layer_idx: int,
        hidden_state: torch.Tensor,
        extended_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Architecture-aware execution of a single Transformer block.
        """
        layer_module = self.model.encoder.layer[layer_idx]
        if position_bias is not None:
            # MPNet architecture
            layer_outputs = layer_module(
                hidden_state, attention_mask=extended_mask, position_bias=position_bias
            )
        else:
            # Standard BERT / MiniLM architecture
            layer_outputs = layer_module(hidden_state, attention_mask=extended_mask)
        out = layer_outputs[0]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out

    @torch.no_grad()
    def encode_step_wise_adaptive(
        self,
        query: str,
        stability_threshold: float = 0.95,
        min_layer: int = 1,
        max_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Executes query encoder layer-by-layer sequentially.
        Stops early when cosine_sim(emb_l, emb_{l-1}) >= stability_threshold.

        Args:
            query: Input query string.
            stability_threshold: Cosine similarity threshold for early termination.
            min_layer: Minimum layer to evaluate before allowing early termination (>= 1).
            max_layer: Maximum layer to evaluate (defaults to self.num_layers).

        Returns:
            Tuple of:
                - final_normalized_embedding: Tensor of shape (1, hidden_dim) on CPU
                - exit_layer: Integer index of stopping layer in [1, num_layers]
                - similarities: List of consecutive cosine similarities [S(2), S(3), ...]
        """
        if max_layer is None or max_layer > self.num_layers:
            max_layer = self.num_layers
        if min_layer < 1:
            min_layer = 1
        if min_layer > max_layer:
            raise ValueError(f"min_layer ({min_layer}) cannot be greater than max_layer ({max_layer})")

        encoded = self.tokenizer(
            [query] if isinstance(query, str) else list(query),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # 1. Compute layer 0 embeddings
        hidden_state = self.model.embeddings(input_ids)

        # 2. Compute extended attention mask
        try:
            extended_mask = self.model.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )
        except Exception:
            extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

        # 3. Check for MPNet position bias
        position_bias = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "compute_position_bias"):
            position_bias = self.model.encoder.compute_position_bias(
                hidden_state, position_ids=None
            )

        prev_emb: Optional[torch.Tensor] = None
        similarities: List[float] = []
        exit_layer: int = max_layer
        final_emb: Optional[torch.Tensor] = None

        for l_idx in range(max_layer):
            current_layer = l_idx + 1

            # Execute single transformer block
            hidden_state = self._forward_transformer_layer(
                l_idx, hidden_state, extended_mask, position_bias
            )

            curr_emb = self.pool_and_normalize(hidden_state, attention_mask)

            if prev_emb is not None:
                # Cosine similarity between unit-normalized consecutive vectors
                sim = float(torch.sum(curr_emb * prev_emb, dim=-1).item())
                similarities.append(sim)

                # Early exit decision
                if current_layer >= min_layer and sim >= stability_threshold:
                    exit_layer = current_layer
                    final_emb = curr_emb
                    break

            prev_emb = curr_emb

            if current_layer == max_layer:
                final_emb = curr_emb
                exit_layer = current_layer
                break

        if final_emb is None:
            final_emb = prev_emb if prev_emb is not None else self.pool_and_normalize(hidden_state, attention_mask)

        return final_emb.cpu(), exit_layer, similarities
