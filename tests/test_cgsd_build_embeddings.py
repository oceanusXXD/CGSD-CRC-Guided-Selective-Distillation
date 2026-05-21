from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.cgsd_build_embeddings import embed_texts_vllm


class BuildEmbeddingsTest(unittest.TestCase):
    def test_embed_texts_vllm_batches_and_normalizes_embeddings(self) -> None:
        class FakeEmbeddingModel:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def embed(self, texts: list[str], *, use_tqdm: bool) -> list[SimpleNamespace]:
                self.calls.append(list(texts))
                outputs = []
                for text in texts:
                    if text == "first":
                        vector = [3.0, 4.0]
                    elif text == "second":
                        vector = [5.0, 0.0]
                    else:
                        raise AssertionError(f"unexpected text: {text}")
                    outputs.append(SimpleNamespace(outputs=SimpleNamespace(embedding=vector)))
                return outputs

        model = FakeEmbeddingModel()
        embeddings = embed_texts_vllm(
            model=model,
            texts=["first", "second"],
            batch_size=1,
        )

        self.assertEqual([["first"], ["second"]], model.calls)
        self.assertEqual(np.float32, embeddings.dtype)
        np.testing.assert_allclose(embeddings, np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
