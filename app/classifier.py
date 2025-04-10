from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

class ConversationClassifier:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Initialize context classification model
        self.context_model_name = "bert-base-uncased"
        self.context_tokenizer = AutoTokenizer.from_pretrained(self.context_model_name)
        self.context_model = AutoModelForSequenceClassification.from_pretrained(
            self.context_model_name,
            num_labels=5  # Adjust based on your context categories
        )
        
        # Define context categories
        self.context_categories = [
            "greeting",
            "technical_support",
            "general_query",
            "complaint",
            "feedback"
        ]

    def classify(self, conversation_text: str) -> dict:
        # Perform sentiment analysis
        sentiment_result = self.sentiment_analyzer(conversation_text)[0]
        sentiment_score = sentiment_result["score"]
        sentiment = "positive" if sentiment_score > 0.5 else "negative"
        
        # Perform context classification
        inputs = self.context_tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.context_model(**inputs)
            context_logits = outputs.logits
            context_probs = torch.softmax(context_logits, dim=1)
            context_idx = torch.argmax(context_probs).item()
            confidence_score = float(context_probs[0][context_idx])
        
        context = self.context_categories[context_idx]
        
        return {
            "sentiment": sentiment,
            "context": context,
            "confidence_score": confidence_score
        } 