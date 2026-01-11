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
        
        # Build context string from relevant entries with clear formatting
        context_parts = []
        for i, entry in enumerate(context_entries, 1):
            topic = entry.get('topic', '').strip()
            text = entry.get('text', '').strip()
            treatment_date = entry.get('treatment date', '').strip()
            treatment_country = entry.get('treatment country', '').strip()
            treatment_given = entry.get('treatment given', '').strip()
            
            # Build a clear context entry
            entry_text = f"[Entry {i}]"
            if topic:
                entry_text += f" Topic: {topic}"
            if text:
                entry_text += f"\nInformation: {text}"
            if treatment_date:
                entry_text += f"\nTreatment Date: {treatment_date}"
            if treatment_country:
                entry_text += f"\nCountry: {treatment_country}"
            if treatment_given:
                entry_text += f"\nTreatment: {treatment_given}"
            
            context_parts.append(entry_text)
        
        context_str = "\n\n".join(context_parts)
        
        # Use chat template format with clear instructions to use dataset
        system_prompt = "You are a medical knowledge assistant. Answer questions using ONLY the information provided in the dataset entries below. Extract and present the relevant facts directly from the dataset. Be clear, accurate, and concise (2-4 sentences). If the dataset doesn't contain the answer, say 'Based on the available dataset, I don't have specific information about this.'"
        user_message = f"Dataset Entries:\n{context_str}\n\nUser Question: {question}\n\nUsing the dataset entries above, provide a clear and accurate answer:"
        
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
                max_new_tokens=200,  # Slightly increased for complete answers
                do_sample=True,  # Sampling for better quality
                temperature=0.5,  # Balanced for clarity and accuracy
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=_tokenizer.eos_token_id,
                early_stopping=True  # Stop when answer is complete
            )
        
        # Decode response
        full_response = _tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response (after <|assistant|>)
        if '<|assistant|>' in full_response:
            answer = full_response.split('<|assistant|>')[-1].strip()
            # Remove eos token if present
            # Remove eos token and other special tokens
            answer = answer.replace('</s>', '').replace('<|user|>', '').replace('<|system|>', '').strip()
            
            # Clean up the answer - remove any repeated phrases or incomplete sentences
            # Split by sentences and keep only complete ones
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            if sentences:
                # Keep sentences that make sense (not too short fragments)
                valid_sentences = [s for s in sentences if len(s) > 10]
                if valid_sentences:
                    answer = '. '.join(valid_sentences[:4])  # Max 4 sentences
                    if not answer.endswith('.'):
                        answer += '.'
                else:
                    answer = '. '.join(sentences[:3])
                    if not answer.endswith('.'):
                        answer += '.'
            
            # Final length check (keep concise but complete)
            if len(answer) > 400:
                sentences = answer.split('. ')
                truncated = []
                for s in sentences:
                    if len('. '.join(truncated + [s])) <= 400:
                        truncated.append(s)
                    else:
                        break
                answer = '. '.join(truncated)
                if answer and not answer.endswith('.'):
                    answer += '.'
            
            return answer if answer and len(answer) > 10 else None
        
        return full_response.strip() if full_response.strip() else None
        
    except Exception as e:
        import traceback
        print(f"Error in answer_with_model: {e}")
        traceback.print_exc()
        return None
