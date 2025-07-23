from typing import Dict
import re
class EmotionAnalyzer:
    """Transformer-based emotion detection"""
    def __init__(self):
        self.emotion_pipeline = None
        self.roberta_pipeline = None
        self._setup_emotion_models()
    
    def _setup_emotion_models(self):
        """Initialize various transformer models for emotion detection"""
        try:
            from transformers import pipeline
            
            # Method 1: Dedicated emotion classification model
            print("🤖 Loading emotion classification model...")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            print("✅ Advanced emotion models loaded successfully!")
            
        except ImportError:
            print("❌ Install transformers: pip install transformers torch")
            self.emotion_pipeline = None
        except Exception as e:
            print(f"⚠️  Error loading emotion models: {e}")
            print("Falling back to keyword-based analysis...")
            self.emotion_pipeline = None

    def analyze_emotions_transformer(self, text: str) -> Dict[str, float]:
        """
        Use transformer models for sophisticated emotion detection
        Returns: Dictionary with emotion scores (0.0 to 1.0)
        """
        if not self.emotion_pipeline or not text.strip():
            return self._fallback_emotion_analysis(text)
        
        try:
            # Clean text for better analysis
            cleaned_text = self._clean_text_for_emotion(text)
            
            # Get emotion predictions
            emotion_results = self.emotion_pipeline(cleaned_text)[0]
            
            # Parse results into our emotion categories
            transformer_emotions = {}
            for result in emotion_results:
                label = result['label'].lower()
                score = result['score']
                transformer_emotions[label] = score
            
            # Map transformer emotions to our skincare-specific categories
            mapped_emotions = self._map_to_skincare_emotions(transformer_emotions, text)
            
            return mapped_emotions
            
        except Exception as e:
            print(f"⚠️  Transformer emotion analysis failed: {e}")
            return self._fallback_emotion_analysis(text)
    
    def _clean_text_for_emotion(self, text: str) -> str:
        """Clean text specifically for emotion analysis"""
        # Preserve emotional indicators while cleaning
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[!]{4,}', '!!!', text)  # Normalize excessive punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long (transformer limits)
        if len(text) > 400:
            text = text[:400] + "..."
        
        return text
    
    def _map_to_skincare_emotions(self, transformer_emotions: Dict[str, float], original_text: str) -> Dict[str, float]:
        """
        Map transformer emotion predictions to skincare-specific emotional categories
        """
        skincare_emotions = {
            "excited": 0.0,
            "frustrated": 0.0,
            "grateful": 0.0,
            "confused": 0.0,
            "satisfied": 0.0,
            "disappointed": 0.0,
            "hopeful": 0.0,
            "desperate": 0.0
        }
        
        text_lower = original_text.lower()
        
        # Excitement mapping
        joy_score = transformer_emotions.get('joy', 0.0)
        surprise_score = transformer_emotions.get('surprise', 0.0)
        
        excitement_base = (joy_score + surprise_score * 0.5)
        
        # Boost if skincare excitement indicators are present
        excitement_boosters = ['holy grail', 'game changer', 'obsessed', 'love', 'amazing', 'incredible']
        boost_factor = 1.0
        for booster in excitement_boosters:
            if booster in text_lower:
                boost_factor += 0.2
        
        skincare_emotions["excited"] = min(1.0, excitement_base * boost_factor)
        
        # Frustration mapping
        anger_score = transformer_emotions.get('anger', 0.0)
        disgust_score = transformer_emotions.get('disgust', 0.0)
        
        frustration_base = (anger_score + disgust_score * 0.7)
        
        # Boost for skincare-specific frustration
        frustration_boosters = ['broke out', 'made worse', 'waste of money', 'doesnt work', 'terrible']
        boost_factor = 1.0
        for booster in frustration_boosters:
            if booster in text_lower:
                boost_factor += 0.3
        
        skincare_emotions["frustrated"] = min(1.0, frustration_base * boost_factor)
        
        # Gratitude (often classified as joy + specific keywords)
        gratitude_base = joy_score * 0.6
        gratitude_keywords = ['thank', 'grateful', 'appreciate', 'blessing', 'saved my skin']
        if any(keyword in text_lower for keyword in gratitude_keywords):
            gratitude_base += 0.4
        
        skincare_emotions["grateful"] = min(1.0, gratitude_base)
        
        # Confusion (often low confidence in multiple emotions + question patterns)
        confusion_base = transformer_emotions.get('surprise', 0.0) * 0.3
        if '?' in original_text:
            confusion_base += 0.2
        
        confusion_keywords = ['confused', 'dont understand', 'which one', 'help me choose']
        if any(keyword in text_lower for keyword in confusion_keywords):
            confusion_base += 0.4
        
        skincare_emotions["confused"] = min(1.0, confusion_base)
        
        # Satisfaction (subdued joy + effectiveness keywords)
        satisfaction_base = joy_score * 0.4
        satisfaction_keywords = ['works well', 'effective', 'good results', 'improvement', 'better']
        if any(keyword in text_lower for keyword in satisfaction_keywords):
            satisfaction_base += 0.5
        
        skincare_emotions["satisfied"] = min(1.0, satisfaction_base)
        
        # Disappointment (sadness + specific context)
        sadness_score = transformer_emotions.get('sadness', 0.0)
        disappointment_base = sadness_score
        
        disappointment_keywords = ['disappointed', 'expected more', 'overhyped', 'let down']
        if any(keyword in text_lower for keyword in disappointment_keywords):
            disappointment_base += 0.3
        
        skincare_emotions["disappointed"] = min(1.0, disappointment_base)
        
        # Hope (low-level joy + future-oriented language)
        hope_base = joy_score * 0.3
        hope_keywords = ['hope', 'fingers crossed', 'maybe this will', 'really want this to work']
        if any(keyword in text_lower for keyword in hope_keywords):
            hope_base += 0.4
        
        skincare_emotions["hopeful"] = min(1.0, hope_base)
        
        # Desperation (fear + sadness + specific skincare context)
        fear_score = transformer_emotions.get('fear', 0.0)
        desperation_base = (fear_score + sadness_score) * 0.5
        
        desperation_keywords = ['desperate', 'tried everything', 'nothing works', 'last resort', 'please help']
        if any(keyword in text_lower for keyword in desperation_keywords):
            desperation_base += 0.5
        
        skincare_emotions["desperate"] = min(1.0, desperation_base)
        
        return skincare_emotions
    
    def _fallback_emotion_analysis(self, text: str) -> Dict[str, float]:
        """
        Fallback keyword-based emotion analysis
        Returns emotion scores (0.0 to 1.0) instead of booleans
        """
        if not text or text.strip() == "":
            return {
                "excited": 0.0, "frustrated": 0.0, "grateful": 0.0, "confused": 0.0,
                "satisfied": 0.0, "disappointed": 0.0, "hopeful": 0.0, "desperate": 0.0
            }
        
        text_lower = text.lower()
        
        emotions = {
            "excited": 0.0,
            "frustrated": 0.0,
            "grateful": 0.0,
            "confused": 0.0,
            "satisfied": 0.0,
            "disappointed": 0.0,
            "hopeful": 0.0,
            "desperate": 0.0
        }
        
        # Enhanced keyword scoring with weights
        emotion_patterns = {
            "excited": [
                ('holy grail', 0.9), ('game changer', 0.9), ('obsessed', 0.8), 
                ('love', 0.7), ('amazing', 0.7), ('incredible', 0.7),
                ('perfect', 0.6), ('transformed my skin', 0.8), ('glowing', 0.6),
                ('flawless', 0.7), ('gorgeous skin', 0.6)
            ],
            "frustrated": [
                ('hate', 0.9), ('terrible', 0.8), ('awful', 0.8),
                ('broke me out', 0.9), ('made worse', 0.8), ('waste of money', 0.7),
                ('doesnt work', 0.7), ("doesn't work", 0.7), ('regret', 0.6),
                ('irritating my skin', 0.8), ('burning', 0.7), ('stinging', 0.7)
            ],
            "grateful": [
                ('thank you', 0.8), ('grateful', 0.9), ('blessing', 0.7),
                ('saved my skin', 0.9), ('appreciate', 0.6), ('thankful', 0.7),
                ('life changing', 0.8), ('miracle worker', 0.8)
            ],
            "confused": [
                ('confused', 0.8), ('dont understand', 0.7), ("don't understand", 0.7),
                ('which one', 0.6), ('help me choose', 0.7), ('not sure', 0.5), 
                ('unclear', 0.6), ('what order', 0.6), ('how often', 0.5),
                ('when to use', 0.5), ('morning or night', 0.5)
            ],
            "satisfied": [
                ('satisfied', 0.7), ('works well', 0.8), ('effective', 0.7),
                ('good results', 0.8), ('improvement', 0.6), ('better', 0.5),
                ('smoother', 0.6), ('softer', 0.6), ('brighter', 0.6), ('cleared up', 0.8)
            ],
            "disappointed": [
                ('disappointed', 0.8), ('expected more', 0.7), ('overhyped', 0.6),
                ('let down', 0.7), ('not worth it', 0.6), ('no difference', 0.7),
                ('no change', 0.6), ('still breaking out', 0.7), ('not what i thought', 0.6)
            ],
            "hopeful": [
                ('hope', 0.6), ('fingers crossed', 0.7), ('maybe this will', 0.5),
                ('really want this to work', 0.8), ('crossing fingers', 0.7),
                ('hopefully', 0.6), ('praying this works', 0.7), ('heard good things', 0.5)
            ],
            "desperate": [
                ('desperate', 0.9), ('tried everything', 0.8), ('nothing works', 0.8),
                ('last resort', 0.9), ('please help', 0.7), ('cant take it anymore', 0.8),
                ("can't take it anymore", 0.8), ('severe acne', 0.7), ('getting worse', 0.6),
                ('painful', 0.6), ('dont know what else', 0.7), ("don't know what else", 0.7)
            ]
        }
        
        # Calculate weighted scores
        for emotion, patterns in emotion_patterns.items():
            max_score = 0.0
            for pattern, weight in patterns:
                if pattern in text_lower:
                    max_score = max(max_score, weight)
            emotions[emotion] = max_score
        
        return emotions
    
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Comprehensive emotion analysis combining transformer and keyword approaches
        """
        # Get transformer-based emotions
        transformer_emotions = self.analyze_emotions_transformer(text)
        
        # Get keyword-based emotions as backup/validation
        keyword_emotions = self._fallback_emotion_analysis(text)
        
        # Combine with weighted average (favor transformer when available)
        combined_emotions = {}
        transformer_weight = 0.7 if self.emotion_pipeline else 0.0
        keyword_weight = 1.0 - transformer_weight
        
        for emotion in transformer_emotions.keys():
            combined_score = (
                transformer_emotions[emotion] * transformer_weight +
                keyword_emotions[emotion] * keyword_weight
            )
            combined_emotions[emotion] = round(combined_score, 3)
        
        return combined_emotions