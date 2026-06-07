"""
UEG Model Loader
Downloads ONNX model + tokenizer from HuggingFace on first startup.
Caches locally so subsequent restarts are instant.
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from config import (
    HF_TOKEN, MODEL_REPO, MAX_SEQ_LEN,
    INTENT_CLASSES, RESOURCE_CLASSES,
)

logger    = logging.getLogger("ueg.loader")
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/ueg-cache"))


class UEGInferenceEngine:
    """
    Wraps ONNX model + tokenizer.
    Thread-safe — onnxruntime sessions are stateless per inference call.
    """

    def __init__(self):
        self.session   = None
        self.tokenizer = None
        self.pad_id    = 0
        self._ready    = False

    def load(self):
        """Download and initialize everything. Called once at startup."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading UEG model from {MODEL_REPO}...")

        # Download ONNX model
        onnx_path = self._download("export/ueg_model.onnx")
        logger.info(f"ONNX model: {onnx_path}")

        # Download tokenizer
        tok_path = self._download("tokenizer/tokenizer.json")
        cfg_path = self._download("tokenizer/tokenizer_config.json")

        # Load tokenizer config for pad_id
        with open(cfg_path) as f:
            tok_cfg = json.load(f)
        self.pad_id = tok_cfg["pad_id"]

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_padding(
            pad_id=self.pad_id,
            pad_token="[PAD]",
            length=MAX_SEQ_LEN,
        )
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        logger.info(f"Tokenizer loaded — vocab: {self.tokenizer.get_vocab_size():,} | pad_id: {self.pad_id}")

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        logger.info(f"ONNX session ready — providers: {self.session.get_providers()}")

        self._ready = True
        logger.info("UEG engine ready")

    def infer(self, text: str) -> dict:
        """
        Run inference on a single text.
        Returns raw logits + probabilities for both heads.
        """
        if not self._ready:
            raise RuntimeError("Engine not loaded — call load() first")

        # Tokenize
        enc  = self.tokenizer.encode(text)
        ids  = enc.ids[:MAX_SEQ_LEN]
        mask = enc.attention_mask[:MAX_SEQ_LEN]

        pad  = MAX_SEQ_LEN - len(ids)
        ids  += [self.pad_id] * pad
        mask += [0] * pad

        np_ids  = np.array([ids],  dtype=np.int64)
        np_mask = np.array([mask], dtype=np.int64)

        # Run ONNX
        outputs = self.session.run(
            ["logits_intent", "logits_resource"],
            {"input_ids": np_ids, "attention_mask": np_mask},
        )

        logits_intent   = outputs[0][0]  # (22,)
        logits_resource = outputs[1][0]  # (5,)

        # Softmax
        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        probs_intent   = softmax(logits_intent)
        probs_resource = softmax(logits_resource)

        intent_idx   = int(np.argmax(probs_intent))
        resource_idx = int(np.argmax(probs_resource))

        return {
            "intent_idx":      intent_idx,
            "resource_idx":    resource_idx,
            "probs_intent":    probs_intent.tolist(),
            "probs_resource":  probs_resource.tolist(),
            "confidence_intent":   float(probs_intent[intent_idx]),
            "confidence_resource": float(probs_resource[resource_idx]),
        }

    def _download(self, filename: str) -> Path:
        """Download file from HF, cache locally."""
        local = CACHE_DIR / filename.replace("/", "_")
        if local.exists():
            logger.debug(f"Cache hit: {filename}")
            return local

        logger.info(f"Downloading: {filename}")
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            repo_type="model",
            
            cache_dir=str(CACHE_DIR / "hf_cache"),
        )
        # Copy to flat cache for easy access
        import shutil
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, local)
        return local

    @property
    def ready(self) -> bool:
        return self._ready


# Singleton — one engine per process
engine = UEGInferenceEngine()
