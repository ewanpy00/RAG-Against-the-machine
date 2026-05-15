"""LLM-based answer generator using Qwen3-0.6B."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.models import Chunk


class AnswerGenerator:
    """Generates answers to questions using a local Qwen3 model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_length: int = 2000,
        max_new_tokens: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.model.eval()
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Generate an answer to a question
          given retrieved chunks as context."""
        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Generation time: {elapsed:.2f}s ({len(new_tokens)} tokens)")

        return answer.strip()

    def _build_context(self, chunks: list[Chunk]) -> str:
        """Concatenate chunk texts into a single context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text[:self.max_context_length]
            parts.append(f"[Source {i}: {chunk.file_path}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Format a chat prompt for the Qwen3 model."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "about the vLLM library based on provided context. "
                    "Use only the information from the context. "
                    "Be concise and accurate. "
                    "If the context does not contain"
                    " the answer, say you don't know. "
                    "Answer should be <= 100 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                ),
            },
        ]

        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(result, str)
        return result
