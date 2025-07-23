import sqlite3
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ContentMetrics:
    """Data class to hold content analysis results"""
    media_id: str
    total_comments: int
    engagement_quality_score: float
    sentiment_distribution: Dict[str, float]  # % positive, negative, neutral
    emotion_distribution: Dict[str, float]   # Top emotions with avg scores
    urgency_breakdown: Dict[str, int]        # Count by urgency level
    lifecycle_breakdown: Dict[str, int]      # Count by customer stage
    viral_indicators: Dict[str, float]       # User tagging, sharing patterns
    content_effectiveness: Dict[str, float]  # Question generation, advice seeking
    top_emotions: List[Tuple[str, float]]    # Top 3 emotions with scores
    actionable_insights: List[str]           # Key takeaways for social media managers


class ContentAnalyzer:
    """
    Comprehensive content analysis for social media posts.
    
    Analyzes engagement patterns, emotional responses, and customer behavior
    at the individual post level to provide actionable content strategy insights.
    """
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize the content analyzer.
        
        Args:
            db_connection: Active SQLite database connection
        """
        self.connection = db_connection
        self.cursor = db_connection.cursor()
    
    def analyze_single_post(self, media_id: str, company_name: str = "@treehut") -> Optional[ContentMetrics]:
        """
        Comprehensive analysis of a single social media post.
        
        Args:
            media_id (str): The media ID to analyze
            company_name (str): Company name to filter by
            
        Returns:
            Optional[ContentMetrics]: Detailed content analysis or None if no data
        """
        # Get basic post information
        post_data = self._get_post_data(media_id, company_name)
        if not post_data:
            return None
        
        total_comments, media_caption, post_timestamp = post_data
        
        # Get all comment analysis data for this post
        comment_data = self._get_comment_analysis_data(media_id, company_name)
        if not comment_data:
            return None
        
        # Calculate comprehensive metrics
        engagement_quality = self._calculate_engagement_quality(comment_data)
        sentiment_dist = self._calculate_sentiment_distribution(comment_data)
        emotion_dist = self._calculate_emotion_distribution(comment_data)
        urgency_breakdown = self._calculate_urgency_breakdown(comment_data)
        lifecycle_breakdown = self._calculate_lifecycle_breakdown(comment_data)
        viral_indicators = self._calculate_viral_indicators(comment_data)
        content_effectiveness = self._calculate_content_effectiveness(comment_data)
        top_emotions = self._get_top_emotions(emotion_dist)
        actionable_insights = self._generate_actionable_insights(
            media_id, sentiment_dist, emotion_dist, urgency_breakdown, 
            lifecycle_breakdown, viral_indicators, content_effectiveness
        )
        
        return ContentMetrics(
            media_id=media_id,
            total_comments=total_comments,
            engagement_quality_score=engagement_quality,
            sentiment_distribution=sentiment_dist,
            emotion_distribution=emotion_dist,
            urgency_breakdown=urgency_breakdown,
            lifecycle_breakdown=lifecycle_breakdown,
            viral_indicators=viral_indicators,
            content_effectiveness=content_effectiveness,
            top_emotions=top_emotions,
            actionable_insights=actionable_insights
        )
    
    def _get_post_data(self, media_id: str, company_name: str) -> Optional[Tuple[int, str, str]]:
        """Get basic post information."""
        self.cursor.execute('''
            SELECT COUNT(*) as total_comments, 
                   MAX(e.media_caption) as caption,
                   MAX(e.timestamp) as latest_timestamp
            FROM engagement e
            JOIN company c ON e.company_id = c.id
            WHERE e.media_id = ? AND c.name = ?
        ''', (media_id, company_name))
        
        result = self.cursor.fetchone()
        if result and result[0] > 0:
            return result
        return None
    
    def _get_comment_analysis_data(self, media_id: str, company_name: str) -> List[Dict]:
        """Get all comment analysis data for a specific post."""
        self.cursor.execute('''
            SELECT ca.word_count, ca.char_count, ca.has_question, ca.has_advice_seeking,
                   ca.emotion_indicators, ca.urgency_level, ca.customer_lifecycle_stage,
                   ca.tags_users, ca.tagged_usernames, ca.sentiment_negative,
                   ca.sentiment_neutral, ca.sentiment_positive, ca.sentiment_label,
                   ca.sentiment_confidence, e.comment_text
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE e.media_id = ? AND c.name = ?
        ''', (media_id, company_name))
        
        columns = [
            'word_count', 'char_count', 'has_question', 'has_advice_seeking',
            'emotion_indicators', 'urgency_level', 'customer_lifecycle_stage',
            'tags_users', 'tagged_usernames', 'sentiment_negative',
            'sentiment_neutral', 'sentiment_positive', 'sentiment_label',
            'sentiment_confidence', 'comment_text'
        ]
        
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def _calculate_engagement_quality(self, comment_data: List[Dict]) -> float:
        """
        Calculate overall engagement quality score (0-1).
        
        Factors:
        - Average comment length (depth of engagement)
        - Question ratio (curiosity/interest)
        - Advice-seeking ratio (active engagement)
        - Sentiment confidence (clear emotional response)
        - User tagging (viral potential)
        """
        if not comment_data:
            return 0.0
        
        total_comments = len(comment_data)
        
        # Average comment length (normalized)
        avg_word_count = sum(c['word_count'] for c in comment_data) / total_comments
        length_score = min(avg_word_count / 20, 1.0)  # Cap at 20 words for full score
        
        # Question engagement ratio
        question_ratio = sum(c['has_question'] for c in comment_data) / total_comments
        
        # Advice-seeking ratio
        advice_ratio = sum(c['has_advice_seeking'] for c in comment_data) / total_comments
        
        # Average sentiment confidence
        avg_confidence = sum(c['sentiment_confidence'] for c in comment_data) / total_comments
        
        # User tagging ratio (viral indicator)
        tagging_ratio = sum(c['tags_users'] for c in comment_data) / total_comments
        
        # Weighted engagement quality score
        quality_score = (
            length_score * 0.25 +           # 25% - Comment depth
            question_ratio * 0.20 +         # 20% - Curiosity/interest
            advice_ratio * 0.20 +           # 20% - Active engagement
            avg_confidence * 0.25 +         # 25% - Clear emotional response
            tagging_ratio * 0.10            # 10% - Viral potential
        )
        
        return round(quality_score, 3)
    
    def _calculate_sentiment_distribution(self, comment_data: List[Dict]) -> Dict[str, float]:
        """Calculate sentiment distribution percentages."""
        if not comment_data:
            return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        total = len(comment_data)
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for comment in comment_data:
            label = comment['sentiment_label']
            if label in sentiment_counts:
                sentiment_counts[label] += 1
        
        return {
            label: round((count / total) * 100, 1)
            for label, count in sentiment_counts.items()
        }
    
    def _calculate_emotion_distribution(self, comment_data: List[Dict]) -> Dict[str, float]:
        """Calculate average emotion scores across all comments."""
        if not comment_data:
            return {}
        
        emotion_totals = {
            "excited": 0.0, "frustrated": 0.0, "grateful": 0.0, "confused": 0.0,
            "satisfied": 0.0, "disappointed": 0.0, "hopeful": 0.0, "desperate": 0.0
        }
        
        valid_emotion_counts = 0
        
        for comment in comment_data:
            try:
                emotions = json.loads(comment['emotion_indicators'])
                for emotion, score in emotions.items():
                    if emotion in emotion_totals and isinstance(score, (int, float)):
                        emotion_totals[emotion] += score
                valid_emotion_counts += 1
            except (json.JSONDecodeError, TypeError):
                continue
        
        if valid_emotion_counts == 0:
            return emotion_totals
        
        return {
            emotion: round(total / valid_emotion_counts, 3)
            for emotion, total in emotion_totals.items()
        }
    
    def _calculate_urgency_breakdown(self, comment_data: List[Dict]) -> Dict[str, int]:
        """Calculate breakdown of urgency levels."""
        urgency_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for comment in comment_data:
            level = comment['urgency_level']
            if level in urgency_counts:
                urgency_counts[level] += 1
        
        return urgency_counts
    
    def _calculate_lifecycle_breakdown(self, comment_data: List[Dict]) -> Dict[str, int]:
        """Calculate breakdown of customer lifecycle stages."""
        lifecycle_counts = {'discovery': 0, 'consideration': 0, 'loyalty': 0}
        
        for comment in comment_data:
            stage = comment['customer_lifecycle_stage']
            if stage in lifecycle_counts:
                lifecycle_counts[stage] += 1
        
        return lifecycle_counts
    
    def _calculate_viral_indicators(self, comment_data: List[Dict]) -> Dict[str, float]:
        """Calculate viral potential indicators."""
        if not comment_data:
            return {'user_tagging_rate': 0.0, 'avg_tags_per_comment': 0.0, 'viral_score': 0.0}
        
        total_comments = len(comment_data)
        comments_with_tags = sum(c['tags_users'] for c in comment_data)
        
        # Count total tags
        total_tags = 0
        for comment in comment_data:
            try:
                tagged_users = json.loads(comment['tagged_usernames'])
                total_tags += len(tagged_users)
            except (json.JSONDecodeError, TypeError):
                continue
        
        user_tagging_rate = (comments_with_tags / total_comments) * 100
        avg_tags_per_comment = total_tags / total_comments if total_comments > 0 else 0
        
        # Viral score combines tagging rate and average tags
        viral_score = (user_tagging_rate / 100) * 0.7 + min(avg_tags_per_comment / 2, 1) * 0.3
        
        return {
            'user_tagging_rate': round(user_tagging_rate, 1),
            'avg_tags_per_comment': round(avg_tags_per_comment, 2),
            'viral_score': round(viral_score, 3)
        }
    
    def _calculate_content_effectiveness(self, comment_data: List[Dict]) -> Dict[str, float]:
        """Calculate content effectiveness metrics."""
        if not comment_data:
            return {'question_generation_rate': 0.0, 'advice_seeking_rate': 0.0, 'effectiveness_score': 0.0}
        
        total_comments = len(comment_data)
        questions = sum(c['has_question'] for c in comment_data)
        advice_seeking = sum(c['has_advice_seeking'] for c in comment_data)
        
        question_rate = (questions / total_comments) * 100
        advice_rate = (advice_seeking / total_comments) * 100
        
        # Effectiveness score combines question generation and advice seeking
        effectiveness_score = (question_rate + advice_rate) / 200  # Normalize to 0-1
        
        return {
            'question_generation_rate': round(question_rate, 1),
            'advice_seeking_rate': round(advice_rate, 1),
            'effectiveness_score': round(effectiveness_score, 3)
        }
    
    def _get_top_emotions(self, emotion_dist: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get top 3 emotions by score."""
        sorted_emotions = sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)
        return [(emotion, score) for emotion, score in sorted_emotions[:3] if score > 0.01]
    
    def _generate_actionable_insights(self, media_id: str, sentiment_dist: Dict[str, float], 
                                    emotion_dist: Dict[str, float], urgency_breakdown: Dict[str, int],
                                    lifecycle_breakdown: Dict[str, int], viral_indicators: Dict[str, float],
                                    content_effectiveness: Dict[str, float]) -> List[str]:
        """Generate actionable insights for social media managers."""
        insights = []
        
        # Sentiment insights
        if sentiment_dist['negative'] > 30:
            insights.append(f"⚠️ High negative sentiment ({sentiment_dist['negative']:.1f}%) - Review content messaging")
        elif sentiment_dist['positive'] > 70:
            insights.append(f"✅ Excellent positive sentiment ({sentiment_dist['positive']:.1f}%) - Replicate content style")
        
        # Emotion insights
        top_emotion = max(emotion_dist.items(), key=lambda x: x[1])
        if top_emotion[1] > 0.3:
            if top_emotion[0] == 'excited':
                insights.append("🎉 High excitement - Perfect for user-generated content campaigns")
            elif top_emotion[0] == 'confused':
                insights.append("💡 High confusion - Create educational follow-up content")
            elif top_emotion[0] == 'frustrated':
                insights.append("🚨 High frustration - Immediate customer service attention needed")
            elif top_emotion[0] == 'grateful':
                insights.append("⭐ High gratitude - Leverage for testimonials and reviews")
        
        # Urgency insights
        if urgency_breakdown['high'] > 5:
            insights.append(f"🚨 {urgency_breakdown['high']} high-urgency comments require immediate response")
        
        # Lifecycle insights
        total_lifecycle = sum(lifecycle_breakdown.values())
        if total_lifecycle > 0:
            discovery_pct = (lifecycle_breakdown['discovery'] / total_lifecycle) * 100
            if discovery_pct > 60:
                insights.append("🔍 High discovery audience - Focus on brand education content")
            elif lifecycle_breakdown['loyalty'] > lifecycle_breakdown['discovery']:
                insights.append("💎 Strong loyalty engagement - Ideal for product launches")
        
        # Viral potential insights
        if viral_indicators['viral_score'] > 0.3:
            insights.append(f"📈 High viral potential ({viral_indicators['user_tagging_rate']:.1f}% tag rate) - Boost promotion")
        
        # Content effectiveness insights
        if content_effectiveness['question_generation_rate'] > 25:
            insights.append("❓ High question generation - Create FAQ or educational series")
        elif content_effectiveness['advice_seeking_rate'] > 20:
            insights.append("🤝 High advice-seeking - Opportunity for expert positioning")
        
        return insights
    
    def analyze_all_posts(self, company_name: str = "@treehut") -> pd.DataFrame:
        """
        Analyze all posts for a company and return summary DataFrame.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            pd.DataFrame: Summary of all post analyses
        """
        # Get all media IDs for the company
        self.cursor.execute('''
            SELECT DISTINCT e.media_id
            FROM engagement e
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            ORDER BY e.media_id
        ''', (company_name,))
        
        media_ids = [row[0] for row in self.cursor.fetchall()]
        
        if not media_ids:
            return pd.DataFrame()
        
        # Analyze each post
        results = []
        for media_id in media_ids:
            metrics = self.analyze_single_post(media_id, company_name)
            if metrics:
                results.append({
                    'media_id': metrics.media_id,
                    'total_comments': metrics.total_comments,
                    'engagement_quality_score': metrics.engagement_quality_score,
                    'positive_sentiment_pct': metrics.sentiment_distribution['positive'],
                    'negative_sentiment_pct': metrics.sentiment_distribution['negative'],
                    'top_emotion': metrics.top_emotions[0][0] if metrics.top_emotions else 'none',
                    'top_emotion_score': metrics.top_emotions[0][1] if metrics.top_emotions else 0.0,
                    'high_urgency_comments': metrics.urgency_breakdown['high'],
                    'loyalty_stage_comments': metrics.lifecycle_breakdown['loyalty'],
                    'viral_score': metrics.viral_indicators['viral_score'],
                    'question_generation_rate': metrics.content_effectiveness['question_generation_rate'],
                    'insights_count': len(metrics.actionable_insights)
                })
        
        return pd.DataFrame(results)
    
    def get_top_performing_posts(self, company_name: str = "@treehut", 
                               metric: str = "engagement_quality_score", 
                               limit: int = 10) -> pd.DataFrame:
        """Get top performing posts by specified metric."""
        df = self.analyze_all_posts(company_name)
        if df.empty:
            return df
        
        return df.nlargest(limit, metric)
    
    def get_content_insights_summary(self, company_name: str = "@treehut") -> Dict:
        """Get high-level content insights summary."""
        df = self.analyze_all_posts(company_name)
        if df.empty:
            return {}
        
        return {
            'total_posts_analyzed': len(df),
            'avg_engagement_quality': df['engagement_quality_score'].mean(),
            'avg_positive_sentiment': df['positive_sentiment_pct'].mean(),
            'total_high_urgency_comments': df['high_urgency_comments'].sum(),
            'posts_with_high_viral_potential': len(df[df['viral_score'] > 0.3]),
            'most_common_top_emotion': df['top_emotion'].mode().iloc[0] if not df['top_emotion'].mode().empty else 'none',
            'posts_driving_questions': len(df[df['question_generation_rate'] > 20])
        }