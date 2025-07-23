from typing import Dict, Tuple, Optional
import re

class SentimentAnalyzer:
    #Transformer-based sentiment analysis for social media comments.
    def __init__(self):
        """Initialize the sentiment analyzer with transformer models."""
        self.sentiment_pipeline = None
        self._setup_sentiment_models()
    
    def _setup_sentiment_models(self):
        """Initialize transformer models for sentiment analysis."""
        try:
            from transformers import pipeline
            
            print("🤖 Loading transformer model for sentiment analysis...")
            
            # Using Twitter-RoBERTa model fine-tuned for social media sentiment
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            print("✅ Sentiment transformer model loaded successfully!")
            
        except ImportError:
            print("❌ Transformers not installed. Please run:")
            print("pip install transformers torch")
            self.sentiment_pipeline = None
        except Exception as e:
            print(f"⚠️  Error loading sentiment transformer model: {e}")
            print("Falling back to keyword-based sentiment analysis...")
            self.sentiment_pipeline = None

    def analyze_sentiment_transformers(self, text: str) -> Tuple[float, float, float, str, float]:
        """
        Analyze sentiment using transformer model.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            Tuple[float, float, float, str, float]: 
                (negative_score, neutral_score, positive_score, label, confidence)
        """
        if not self.sentiment_pipeline or not text or text.strip() == "":
            return 0.0, 1.0, 0.0, 'neutral', 0.5
        
        try:
            # Clean text for better analysis
            cleaned_text = self._clean_text_for_analysis(text)
            
            # Get sentiment scores
            results = self.sentiment_pipeline(cleaned_text)[0]
            
            # Parse results - the model returns all scores
            scores = {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0}
            max_score = 0.0
            predicted_label = 'neutral'
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Map model labels to our labels
                if label in ['negative', 'neg']:
                    scores['negative'] = score
                elif label in ['positive', 'pos']:  
                    scores['positive'] = score
                elif label in ['neutral', 'neu']:
                    scores['neutral'] = score
                
                # Track highest confidence prediction
                if score > max_score:
                    max_score = score
                    predicted_label = label
            
            # Ensure we have valid scores
            negative_score = scores.get('negative', 0.0)
            neutral_score = scores.get('neutral', 0.0)
            positive_score = scores.get('positive', 0.0)
            
            # Map label names consistently
            if predicted_label in ['neg', 'negative']:
                final_label = 'negative'
            elif predicted_label in ['pos', 'positive']:
                final_label = 'positive'
            else:
                final_label = 'neutral'
            
            confidence = max_score
            
            return (
                round(negative_score, 3),
                round(neutral_score, 3), 
                round(positive_score, 3),
                final_label,
                round(confidence, 3)
            )
            
        except Exception as e:
            print(f"⚠️  Transformer sentiment analysis failed for text: {text[:50]}...")
            print(f"Error: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """
        Clean text for better sentiment analysis.
        
        Args:
            text (str): Raw comment text
            
        Returns:
            str: Cleaned text optimized for sentiment analysis
        """
        # Remove excessive punctuation but keep some for context
        text = re.sub(r'[!]{3,}', '!!', text)  # Reduce multiple ! to !!
        text = re.sub(r'[?]{2,}', '?', text)   # Reduce multiple ? to single ?
        
        # Remove @mentions for sentiment (but preserve for other analysis)
        text = re.sub(r'@\w+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate very long texts (transformers have token limits)
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text
    
    def _fallback_sentiment_analysis(self, text: str) -> Tuple[float, float, float, str, float]:
        """
        Fallback keyword-based sentiment analysis when transformers aren't available.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            Tuple[float, float, float, str, float]: 
                (negative_score, neutral_score, positive_score, label, confidence)
        """
        if not text or text.strip() == "":
            return 0.0, 1.0, 0.0, 'neutral', 0.5
        
        text_lower = text.lower()
        
        # Skincare-specific positive words with weights
        positive_words = {
            # High positive impact
            'love': 3, 'amazing': 3, 'perfect': 3, 'incredible': 3, 'holy grail': 4,
            'game changer': 4, 'obsessed': 3, 'addiction': 3, 'miracle': 4,
            # Medium positive impact
            'works': 2, 'effective': 2, 'good': 2, 'great': 2, 'nice': 2,
            'cleared': 3, 'improved': 2, 'glowing': 2, 'smooth': 2, 'soft': 2,
            'recommend': 2, 'thank': 2, 'grateful': 3, 'appreciate': 2,
            # Low positive impact
            'okay': 1, 'fine': 1, 'decent': 1, 'better': 1
        }
        
        # Skincare-specific negative words with weights  
        negative_words = {
            # High negative impact
            'terrible': -4, 'awful': -4, 'hate': -4, 'worst': -4, 'horrible': -4,
            'broke out': -4, 'breakout': -3, 'allergic': -4, 'burned': -4,
            # Medium negative impact
            'disappointed': -2, 'frustrated': -2, 'irritated': -3, 'annoyed': -2,
            'doesnt work': -3, "doesn't work": -3, 'made worse': -3, 'regret': -2,
            'waste': -3, 'money': -1,  # "waste of money" gets both
            # Low negative impact
            'meh': -1, 'okay': -1, 'not great': -2
        }
        
        # Calculate weighted sentiment score
        positive_score = 0
        negative_score = 0
        total_words = len(text_lower.split())
        
        for word, weight in positive_words.items():
            if word in text_lower:
                positive_score += weight
        
        for word, weight in negative_words.items():
            if word in text_lower:
                negative_score += abs(weight)  # Make positive for scoring
        
        # Normalize scores
        max_possible = max(positive_score + negative_score, 1)
        norm_positive = positive_score / max_possible
        norm_negative = negative_score / max_possible
        norm_neutral = max(0, 1 - norm_positive - norm_negative)
        
        # Determine final sentiment
        if norm_positive > norm_negative and norm_positive > 0.3:
            final_label = 'positive'
            confidence = norm_positive
        elif norm_negative > norm_positive and norm_negative > 0.3:
            final_label = 'negative' 
            confidence = norm_negative
        else:
            final_label = 'neutral'
            confidence = norm_neutral
        
        return (
            round(norm_negative, 3),
            round(norm_neutral, 3),
            round(norm_positive, 3),
            final_label,
            round(confidence, 3)
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Comprehensive sentiment analysis with detailed breakdown.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            Dict[str, float]: Detailed sentiment scores and metadata
        """
        neg_score, neu_score, pos_score, label, confidence = self.analyze_sentiment_transformers(text)
        
        return {
            'negative': neg_score,
            'neutral': neu_score,
            'positive': pos_score,
            'label': label,
            'confidence': confidence,
            'compound': pos_score - neg_score  # Overall sentiment score (-1 to 1)
        }
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get just the sentiment label for quick classification.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            str: 'positive', 'negative', or 'neutral'
        """
        _, _, _, label, _ = self.analyze_sentiment_transformers(text)
        return label
    
    def is_positive(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text has positive sentiment above threshold.
        
        Args:
            text (str): Comment text to analyze
            threshold (float): Minimum confidence for positive classification
            
        Returns:
            bool: True if positive sentiment above threshold
        """
        _, _, pos_score, label, confidence = self.analyze_sentiment_transformers(text)
        return label == 'positive' and confidence >= threshold
    
    def is_negative(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text has negative sentiment above threshold.
        
        Args:
            text (str): Comment text to analyze
            threshold (float): Minimum confidence for negative classification
            
        Returns:
            bool: True if negative sentiment above threshold
        """
        neg_score, _, _, label, confidence = self.analyze_sentiment_transformers(text)
        return label == 'negative' and confidence >= threshold
    
    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for sentiment prediction.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        _, _, _, _, confidence = self.analyze_sentiment_transformers(text)
        return confidence
    
    def batch_analyze(self, texts: list) -> list:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Args:
            texts (list): List of comment texts to analyze
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results

