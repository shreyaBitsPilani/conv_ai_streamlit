# slm_generation.py
"""
Small Language Model (SLM) for final response generation using an open-source model (e.g. T5-Small).
No proprietary APIs are used.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class SLMResponseGenerator:
    def __init__(self, model_name="t5-small"):
        """
        Initializes a T5-Small model and tokenizer.
        You can switch to other open-source models, e.g. t5-base, bart-base, etc.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate_response(self, query, retrieved_docs, max_length=128):
        """
        Given the user's query and a list of retrieved_docs (which contain text/context),
        format a prompt for T5, run inference, and return the generated answer.
        """
        # Concatenate top N retrieved docs as context
        # (Adjust N, prompt format, etc. to your liking)
        context = ""
        for i, doc in enumerate(retrieved_docs[:3]):
            context += f"Document {i+1}: {doc['text']}\n"
        
        # Construct an input prompt for T5. 
        # A simple approach:
        prompt = f"question: {query}  context: {context}  "
        
        # Tokenize & generate
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            num_beams=4, 
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
