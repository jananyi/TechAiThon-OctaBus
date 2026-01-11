"""Local TinyLlama model integration for answering questions using the global dataset as reference.

This module provides an `answer_with_model(question, context_entries)` function
that uses TinyLlama-1.1B-Chat-v1.0 to generate answers based on the provided dataset context.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'pipeline' / 'llama_config.json'

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if not CFG.exists():
        raise RuntimeError('llama_config.json not found')
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    model_name = cfg.get('model_name')
    if not model_name:
        raise RuntimeError('model_name not set in config')
    try:
        # lazy import to avoid heavy dependency unless needed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token if not already set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            _model.to('cuda')
        else:
            _model.to('cpu')
        _model.eval()
    except Exception as e:
        raise RuntimeError(f'Failed to load model: {e}')


def answer_with_model(question: str, context_entries: list) -> str:
    """
    Generate an answer using TinyLlama model based on the dataset context.
    
    Args:
        question: The user's question
        context_entries: List of relevant entries from global_kb.jsonl (dicts with topic, text, etc.)
    
    Returns:
        Generated answer string, or None on failure
    """
    try:
        if _model is None:
            _load_model()
        
        # Build context string from relevant entries
        context_parts = []
        for entry in context_entries:
            topic = entry.get('topic', '').strip()
            text = entry.get('text', '').strip()
            if topic and text:
                context_parts.append(f"Topic: {topic}\n{text}")
            elif text:
                context_parts.append(text)
        
        context_str = "\n\n".join(context_parts)
        
        # Use chat template format for TinyLlama-Chat model
        system_prompt = "You are a helpful assistant. Answer the user's question based ONLY on the provided context from the knowledge base. If the context doesn't contain relevant information, say so. Do not make up information."
        user_message = f"Context from knowledge base:\n{context_str}\n\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and generate
        inputs = _tokenizer(prompt, return_tensors='pt')
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = _tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response (after <|assistant|>)
        if '<|assistant|>' in full_response:
            answer = full_response.split('<|assistant|>')[-1].strip()
            # Remove eos token if present
            answer = answer.replace('</s>', '').strip()
            return answer if answer else None
        
        return full_response.strip() if full_response.strip() else None
        
    except Exception as e:
        import traceback
        print(f"Error in answer_with_model: {e}")
        traceback.print_exc()
        return None
