"""LLM-based answer generator using Qwen3-0.6B."""

import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.models import Chunk


class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_length: int = 2000,
        max_new_tokens: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
    
    def generate(
        self,
        question: str,
        chunks: List[Chunk],
    ) -> str:
        context = self._build_context(chunks)
        
        prompt = self._build_prompt(question, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def _build_context(self, chunks: List[Chunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text[:self.max_context_length]
            context_parts.append(
                f"[Source {i}: {chunk.file_path}]\n{text}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "about the vLLM library based on provided context. "
                    "Use only the information from the context. "
                    "Be concise and accurate."
                    " If the context does not contain the answer, say you don't know."
                    " Answer should be <= 100 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer:"
                ),
            },
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )