from pathlib import Path
from typing import List
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from student.models import Chunk


class EmbeddingIndexer:
    """Build dense embedding index for chunks."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.batch_size = batch_size

        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def build_index(
        self,
        chunks: List[Chunk],
        output_dir: Path,
    ) -> None:
        """Compute embeddings for all chunks and save to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        texts = [chunk.text for chunk in chunks]

        print(f"Computing embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        metadata = {
            "model_name": self.model._first_module().auto_model.config.name_or_path,
            "embedding_dim": embeddings.shape[1],
            "num_chunks": len(chunks),
        }
        metadata_path = output_dir / "embeddings_meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Shape: {embeddings.shape}")
        print(f"Disk size: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB")