#!/usr/bin/env python3
"""
Min-K% Prob: A pretraining data detection method.
Based on the hypothesis that unseen examples contain outlier words with low probabilities.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MinKProbDetector:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the Min-K% Prob detector."""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_token_probabilities(self, text: str) -> List[float]:
        """Evaluate token probabilities in the text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get probabilities for each token
        probs = torch.softmax(logits, dim=-1)
        
        # Get the probability of each actual token in the sequence
        token_probs = []
        for i in range(inputs.input_ids.shape[1] - 1):  # -1 because we predict next token
            token_id = inputs.input_ids[0, i + 1]  # Next token
            prob = probs[0, i, token_id].item()
            token_probs.append(prob)
        
        return token_probs
    
    def get_min_k_percent_tokens(self, token_probs: List[float], k_percent: float = 20.0) -> List[float]:
        """Pick the k% tokens with minimum probabilities."""
        if not token_probs:
            return []
        
        k_count = max(1, int(len(token_probs) * k_percent / 100))
        sorted_probs = sorted(token_probs)
        return sorted_probs[:k_count]
    
    def compute_average_log_likelihood(self, min_k_probs: List[float]) -> float:
        """Compute average log likelihood of the minimum k% tokens."""
        if not min_k_probs:
            return float('-inf')
        
        # Convert to log probabilities and compute average
        log_probs = [np.log(max(prob, 1e-10)) for prob in min_k_probs]  # Avoid log(0)
        return np.mean(log_probs)
    
    def detect_pretraining_data(self, text: str, k_percent: float = 20.0, threshold: float = -5.0) -> Dict:
        """
        Detect if text was in LLM's pretraining data using Min-K% Prob method.
        
        Args:
            text: Input text to analyze
            k_percent: Percentage of minimum probability tokens to consider
            threshold: Log likelihood threshold for detection
            
        Returns:
            Dict with detection results
        """
        # Step 1: Evaluate token probabilities
        token_probs = self.get_token_probabilities(text)
        
        # Step 2: Pick k% tokens with minimum probabilities
        min_k_probs = self.get_min_k_percent_tokens(token_probs, k_percent)
        
        # Step 3: Compute average log likelihood
        avg_log_likelihood = self.compute_average_log_likelihood(min_k_probs)
        
        # Detection: High average log likelihood → likely in pretraining data
        is_pretraining_data = avg_log_likelihood > threshold
        
        return {
            "text": text,
            "total_tokens": len(token_probs),
            "k_percent": k_percent,
            "min_k_tokens_count": len(min_k_probs),
            "avg_log_likelihood": avg_log_likelihood,
            "threshold": threshold,
            "is_pretraining_data": is_pretraining_data,
            "confidence": avg_log_likelihood - threshold
        }
    
    def analyze_dataset(self, json_file: str, k_percent: float = 20.0, threshold: float = -5.0) -> List[Dict]:
        """Analyze a dataset saved by the data loader."""
        with open(json_file, 'r') as f:
            samples = json.load(f)
        
        results = []
        for sample in samples:
            result = self.detect_pretraining_data(sample["data"], k_percent, threshold)
            result["sample_id"] = sample["id"]
            results.append(result)
            
        return results

def main():
    """Example usage of Min-K% Prob detector."""
    # Initialize detector
    detector = MinKProbDetector("gpt2")  # Replace with your model
    
    # Example text analysis
    text = "The quick brown fox jumps over the lazy dog."
    result = detector.detect_pretraining_data(text)
    
    print("Min-K% Prob Detection Results:")
    print(f"Text: {result['text']}")
    print(f"Average Log Likelihood: {result['avg_log_likelihood']:.4f}")
    print(f"Is Pretraining Data: {result['is_pretraining_data']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Analyze dataset (if you have the JSON file from data loader)
    # results = detector.analyze_dataset("CodeAlpaca-20k_1000.json")
    # for result in results[:5]:  # Show first 5 results
    #     print(f"Sample {result['sample_id']}: {result['is_pretraining_data']}")

if __name__ == "__main__":
    main() 