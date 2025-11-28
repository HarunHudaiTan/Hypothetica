#!/usr/bin/env python3
"""
embed_mvp.py
Minimal, practical embedding pipeline with:
- Two interchangeable backends: OpenAI & SentenceTransformers
- E5-style query/passages handling (good for academic search)
- FAISS vector index (+ cosine similarity via inner product after L2-normalization)
- On-disk persistence for index + metadata
- Lightweight SQLite cache to avoid re-embedding duplicates across runs
- Simple CLI: build / query

Usage examples
-------------
# Build an index from JSONL (fields: id, title, abstract, url, year, categories)
python embed_mvp.py build sample_papers.jsonl --backend st --model intfloat/e5-base-v2 --out ./index_dir

# Query
python embed_mvp.py query "LLM-based novelty detection for literature review" --backend st --model intfloat/e5-base-v2 --index ./index_dir --topk 5

# Using OpenAI (requires OPENAI_API_KEY in env)
python embed_mvp.py build sample_papers.jsonl --backend openai --model text-embedding-3-large --out ./index_dir
python embed_mvp.py query "graph contrastive learning for drug discovery" --backend openai --model text-embedding-3-large --index ./index_dir
"""
import pickle
import hashlib
import argparse
import os
import sys
import json
import math
import sqlite3
import time
import functools
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Tuple, Optional

# --------- Optional heavy deps are imported lazily ----------
# We'll import faiss and sentence_transformers only when needed to keep import errors readable.


@dataclass
class PaperDoc:
    id: str
    title: str
    abstract: str
    url: Optional[str] = None
    year: Optional[int] = None
    categories: Optional[List[str]] = None

    def text_for_embedding(self) -> str:
        # Passage text (E5 style): we don't add "passage:" token to keep generic.
        # If using e5 models, we will inject "passage:" in the backend itself.
        parts = [self.title or "", self.abstract or ""]
        return "\n\n".join([p.strip() for p in parts if p and p.strip()])


# ---------------------- Cache ----------------------
class EmbedCache:
    """Tiny SQLite cache: key = sha256(model_name + '\n' + text), value = bytes(vector)"""
    def __init__(self, path: str):
        self.path = path
        self._ensure()

    def _ensure(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cache(
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                vec  BLOB NOT NULL
            )
            """)
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: List[str]) -> Dict[int, bytes]:
        out = {}
        if not texts:
            return out
        keys = [(self._key(model, t), i) for i, t in enumerate(texts)]
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            found = {}
            # Fetch in chunks to avoid SQLite variable limit
            CHUNK = 500
            for off in range(0, len(keys), CHUNK):
                sub = keys[off:off+CHUNK]
                ks = [k for k,_ in sub]
                qmarks = ",".join("?"*len(ks))
                cur.execute(f"SELECT key, vec FROM cache WHERE key IN ({qmarks})", ks)
                for k, vec in cur.fetchall():
                    found[k] = vec
            for (k, idx) in keys:
                if k in found:
                    out[idx] = found[k]
        finally:
            con.close()
        return out

    def put_many(self, model: str, pairs: List[Tuple[str, bytes]]):
        if not pairs:
            return
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            cur.executemany("INSERT OR REPLACE INTO cache(key, model, vec) VALUES(?,?,?)",
                            [(self._key(model, txt), model, vec) for txt, vec in pairs])
            con.commit()
        finally:
            con.close()


# ---------------------- Backends ----------------------
class EmbeddingBackend:
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class STBackend(EmbeddingBackend):
    """SentenceTransformers backend (e.g., E5/BGE)."""
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64):
        self._model_name = model_name
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        # Determine if we should use E5 query/passages prefixes
        self.use_e5_prefix = 'e5' in model_name.lower()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # Import torch lazily to ensure env friendliness
        import numpy as np
        embs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).astype("float32")
        return embs.tolist()

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        if self.use_e5_prefix:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        if self.use_e5_prefix:
            texts = [f"query: {t}" for t in texts]
        return self._encode(texts)


class OpenAIBackend(EmbeddingBackend):
    """OpenAI embeddings backend (text-embedding-3-large/small, etc.)."""
    def __init__(self, model_name: str):
        self._model_name = model_name
        try:
            import openai  # legacy
        except Exception:
            pass
        from openai import OpenAI
        self.client = OpenAI()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Compact batching to respect rate limits
        out = []
        B = 2048  # safe-ish chunk size in tokens/characters is model-dependent; we keep per-call lists modest
        for i in range(0, len(texts), B):
            chunk = texts[i:i+B]
            resp = self.client.embeddings.create(model=self._model_name, input=chunk)
            for d in resp.data:
                out.append(d.embedding)
            time.sleep(0.05)  # gentle pacing
        return out

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)


# ---------------------- FAISS Indexer ----------------------
class FaissIndexer:
    def __init__(self, dim: int, workdir: str):
        self.dim = dim
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.index_path = os.path.join(workdir, "index.faiss")
        self.meta_path = os.path.join(workdir, "meta.jsonl")
        self.dim_path  = os.path.join(workdir, "dim.txt")

        self.index = None
        self._ensure_index()

    def _ensure_index(self):
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("faiss is required. pip install faiss-cpu") from e

        import faiss
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.dim_path, "r") as f:
                d = int(f.read().strip())
            if d != self.index.d:
                raise RuntimeError(f"Index dim mismatch: {d} vs {self.index.d}")
        else:
            # Cosine via inner product with normalized vectors
            self.index = faiss.IndexFlatIP(self.dim)
            with open(self.dim_path, "w") as f:
                f.write(str(self.dim))

    def add(self, vectors, metas: List[Dict[str, Any]]):
        import faiss
        import numpy as np
        # Ensure float32 and L2-normalized
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)
        # Append metadata as JSONL
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def save(self):
        import faiss
        faiss.write_index(self.index, self.index_path)

    def search(self, query_vecs, topk=10) -> List[List[Tuple[int, float]]]:
        import faiss
        import numpy as np
        q = np.array(query_vecs, dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, topk)
        out = []
        for row_idx, sc_row in enumerate(scores):
            out.append([(int(idxs[row_idx, j]), float(sc_row[j])) for j in range(sc_row.shape[0])])
        return out

    def iter_meta(self) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.meta_path):
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ---------------------- Pipeline ----------------------
class EmbedPipeline:
    def __init__(self, backend: EmbeddingBackend, cache_path: str, out_dir: str):
        self.backend = backend
        self.cache = EmbedCache(cache_path)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _embed_with_cache(self, texts: List[str], is_query: bool) -> List[List[float]]:
        model = self.backend.model_name
        found = self.cache.get_many(model, texts)
        to_compute = []
        order = [None]*len(texts)

        for i, t in enumerate(texts):
            if i in found:
                order[i] = found[i]
            else:
                to_compute.append((i, t))

        # Compute for missing
        missing_texts = [t for _, t in to_compute]
        if missing_texts:
            embs = self.backend.embed_queries(missing_texts) if is_query else self.backend.embed_passages(missing_texts)
            serialized = []
            for (i, _), vec in zip(to_compute, embs):
                order[i] = vec
                serialized.append((texts[i], memoryview(bytearray(memoryview(bytes(bytearray()))))))
            # Actually store as pickle to preserve float array
            to_store = []
            for (i, txt), vec in zip(to_compute, embs):
                to_store.append((txt, pickle.dumps(vec)))
            self.cache.put_many(model, to_store)

        # Deserialize any cached blobs
        out = []
        for i in range(len(texts)):
            v = order[i]
            if isinstance(v, (bytes, bytearray, memoryview)):
                v = pickle.loads(bytes(v))
            out.append(v)
        return out

    def build(self, docs: List[PaperDoc]) -> Tuple[FaissIndexer, int]:
        # Compute passage embeddings for all docs
        texts = [d.text_for_embedding() for d in docs]
        embs = self._embed_with_cache(texts, is_query=False)
        dim = len(embs[0]) if embs else 0
        if dim <= 0:
            raise RuntimeError("Failed to produce embeddings.")
        indexer = FaissIndexer(dim=dim, workdir=self.out_dir)
        metas = [asdict(d) for d in docs]
        indexer.add(embs, metas)
        indexer.save()
        return indexer, len(docs)

    def query(self, index_dir: str, queries: List[str], topk=10):
        # Create a throwaway indexer (it will load existing files)
        # We need the dim, so embed a dummy to detect dimension; but better: read dim.txt
        dim_path = os.path.join(index_dir, "dim.txt")
        if not os.path.exists(dim_path):
            raise RuntimeError("Invalid index dir; dim.txt not found.")
        with open(dim_path, "r") as f:
            dim = int(f.read().strip())
        idx = FaissIndexer(dim=dim, workdir=index_dir)
        q_embs = self._embed_with_cache(queries, is_query=True)
        results = idx.search(q_embs, topk=topk)
        # Materialize metadata into a list for fast lookup
        metas = list(idx.iter_meta())
        out = []
        for qi, res in enumerate(results):
            rows = []
            for (rid, score) in res:
                if 0 <= rid < len(metas):
                    rows.append({"rank": len(rows)+1, "score": score, **metas[rid]})
            out.append(rows)
        return out


# ---------------------- Utilities ----------------------
def load_jsonl(path: str) -> List[PaperDoc]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(PaperDoc(
                id=str(obj.get("id", "")),
                title=obj.get("title", ""),
                abstract=obj.get("abstract", ""),
                url=obj.get("url"),
                year=obj.get("year"),
                categories=obj.get("categories"),
            ))
    return docs


# ---------------------- CLI ----------------------
def main():
    p = argparse.ArgumentParser(description="Embedding + FAISS MVP")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build index from JSONL")
    pb.add_argument("jsonl", type=str, help="Input JSONL with {id,title,abstract,...}")
    pb.add_argument("--backend", choices=["st","openai"], default="st")
    pb.add_argument("--model", type=str, default="intfloat/e5-base-v2")
    pb.add_argument("--cache", type=str, default="./.embed_cache/cache.sqlite3")
    pb.add_argument("--out", type=str, default="./index_dir")
    pb.add_argument("--device", type=str, default=None, help="SentenceTransformers device, e.g., cuda or mps")

    pq = sub.add_parser("query", help="Query an existing index")
    pq.add_argument("text", type=str, help="Query text")
    pq.add_argument("--backend", choices=["st","openai"], default="st")
    pq.add_argument("--model", type=str, default="intfloat/e5-base-v2")
    pq.add_argument("--cache", type=str, default="./.embed_cache/cache.sqlite3")
    pq.add_argument("--index", type=str, default="./index_dir")
    pq.add_argument("--topk", type=int, default=10)
    pq.add_argument("--device", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "build":
        if args.backend == "st":
            backend = STBackend(args.model, device=args.device)
        else:
            backend = OpenAIBackend(args.model)
        pipeline = EmbedPipeline(backend, cache_path=args.cache, out_dir=args.out)
        docs = load_jsonl(args.jsonl)
        idx, n = pipeline.build(docs)
        print(f"Built index with {n} docs at: {args.out}")

    elif args.cmd == "query":
        if args.backend == "st":
            backend = STBackend(args.model, device=args.device)
        else:
            backend = OpenAIBackend(args.model)
        pipeline = EmbedPipeline(backend, cache_path=args.cache, out_dir=args.index)
        res = pipeline.query(index_dir=args.index, queries=[args.text], topk=args.topk)
        print(json.dumps(res[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
