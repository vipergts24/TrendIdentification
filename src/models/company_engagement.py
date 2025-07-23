import sqlite3
import pandas as pd
import os 
import re
import json
import time
from typing import Dict, Tuple, List, Optional
from dataclasses import  dataclass
from models.emotion_analyzer import EmotionAnalyzer
from models.sentiment_analyzer import SentimentAnalyzer
from models.content_analyzer import ContentAnalyzer,ContentMetrics

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not installed. Install with: pip install plotly")

@dataclass
class CommentMetrics:
    """Data class to hold comment analysis results"""
    word_count: int
    char_count: int
    has_question: bool
    has_advice_seeking: bool
    emotion_indicators: str  # JSON string of emotions detected
    urgency_level: str  # low/medium/high
    customer_lifecycle_stage: str  # discovery/consideration/loyalty
    tags_users: bool
    tagged_usernames: str  # JSON string of @usernames
    sentiment_negative: float
    sentiment_neutral: float
    sentiment_positive: float
    sentiment_label: str
    sentiment_confidence: float

class CompanyEngagement:
    def __init__(self, name: str = "@treehut", type: str = "skincare", db_path: str = "../data/engagements.db"):
        """Initialize Company Engagement with specialized analyzers"""
        self.name = name
        self.type = type
        
        # Handle database path - create absolute path and ensure directory exists
        if not os.path.isabs(db_path):
            self.db_path = os.path.abspath(db_path)
        else:
            self.db_path = db_path
            
        # Ensure the directory exists
        db_directory = os.path.dirname(self.db_path)
        if db_directory and not os.path.exists(db_directory):
            os.makedirs(db_directory, exist_ok=True)
            print(f"📁 Created directory: {db_directory}")
        
        self.connection = None 
        
        # Initialize our analyzers
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        self.content_analyzer = None

        # Color schemes for visualizations
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff9800',
            'error': '#d62728',
            'info': '#17a2b8',
            'sentiment': {
                'positive': '#2ca02c',
                'neutral': '#ffc107',
                'negative': '#dc3545'
            },
            'emotions': {
                'excited': '#ff6b6b',
                'frustrated': '#ee5a24',
                'grateful': '#00d2d3',
                'confused': '#fd9644',
                'satisfied': '#26de81',
                'disappointed': '#a55eea',
                'hopeful': '#45aaf2',
                'desperate': '#fc5c65'
            },
            'urgency': {
                'low': '#28a745',
                'medium': '#ffc107', 
                'high': '#dc3545'
            },
            'lifecycle': {
                'discovery': '#17a2b8',
                'consideration': '#fd7e14',
                'loyalty': '#6f42c1'
            }
        }

        print(f"🗄️  Database path: {self.db_path}")
        
        self._ensure_connection()
        self._setup_analyzers()
        
    def _ensure_connection(self) -> sqlite3.Connection:
        """Ensure database connection is active with better error handling"""
        if self.connection is None:
            try:
                self.connection = sqlite3.connect(self.db_path)
                print(f"✅ Connected to database: {self.db_path}")
            except sqlite3.OperationalError as e:
                print(f"❌ Database connection error: {e}")
                print(f"📍 Attempted path: {self.db_path}")
                print(f"📍 Current working directory: {os.getcwd()}")
                print(f"📍 Path exists: {os.path.exists(os.path.dirname(self.db_path))}")
                raise
        return self.connection
    
    def _setup_analyzers(self):
        """Initialize the specialized sentiment, emotion, and content analyzers"""
        print("🧠 Initializing specialized analyzers...")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize emotion analyzer
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Initialize content analyzer (will be created when needed)
        self.content_analyzer = None
        
        print("✅ All analyzers initialized successfully!")
    
    def _get_content_analyzer(self) -> ContentAnalyzer:
        """Get or create content analyzer instance."""
        if self.content_analyzer is None:
            self.content_analyzer = ContentAnalyzer(self._ensure_connection())
        return self.content_analyzer
    
    def analyze_post_performance(self, media_id: str) -> Optional[ContentMetrics]:
        """
        Analyze performance of a specific post.
        
        Args:
            media_id (str): The media ID to analyze
            
        Returns:
            Optional[ContentMetrics]: Detailed post analysis or None if no data
        """
        content_analyzer = self._get_content_analyzer()
        return content_analyzer.analyze_single_post(media_id, self.name)
    
    def get_all_posts_summary(self) -> pd.DataFrame:
        """
        Get summary analysis of all posts.
        
        Returns:
            pd.DataFrame: Summary of all post analyses
        """
        content_analyzer = self._get_content_analyzer()
        return content_analyzer.analyze_all_posts(self.name)
    
    def get_top_performing_posts(self, metric: str = "engagement_quality_score", limit: int = 10) -> pd.DataFrame:
        """
        Get top performing posts by specified metric.
        
        Args:
            metric (str): Metric to sort by (engagement_quality_score, viral_score, etc.)
            limit (int): Number of top posts to return
            
        Returns:
            pd.DataFrame: Top performing posts
        """
        content_analyzer = self._get_content_analyzer()
        return content_analyzer.get_top_performing_posts(self.name, metric, limit)
    
    def get_content_insights_summary(self) -> Dict:
        """
        Get high-level content insights summary.
        
        Returns:
            Dict: Summary of content performance metrics
        """
        content_analyzer = self._get_content_analyzer()
        return content_analyzer.get_content_insights_summary(self.name)
    
    def analyze_content_trends(self) -> Dict:
        """
        Analyze content trends and patterns.
        
        Returns:
            Dict: Content trend analysis
        """
        df = self.get_all_posts_summary()
        if df.empty:
            return {"error": "No posts to analyze"}
        
        # Calculate trend metrics
        trends = {
            "total_posts": len(df),
            "average_metrics": {
                "engagement_quality": round(df['engagement_quality_score'].mean(), 3),
                "positive_sentiment": round(df['positive_sentiment_pct'].mean(), 1),
                "viral_potential": round(df['viral_score'].mean(), 3),
                "question_generation": round(df['question_generation_rate'].mean(), 1)
            },
            "top_performers": {
                "highest_engagement": df.loc[df['engagement_quality_score'].idxmax(), 'media_id'],
                "most_positive": df.loc[df['positive_sentiment_pct'].idxmax(), 'media_id'],
                "highest_viral": df.loc[df['viral_score'].idxmax(), 'media_id']
            },
            "emotion_patterns": df['top_emotion'].value_counts().to_dict(),
            "urgency_hotspots": df[df['high_urgency_comments'] > 0]['media_id'].tolist(),
            "performance_distribution": {
                "high_quality": len(df[df['engagement_quality_score'] > 0.7]),
                "medium_quality": len(df[(df['engagement_quality_score'] > 0.4) & (df['engagement_quality_score'] <= 0.7)]),
                "low_quality": len(df[df['engagement_quality_score'] <= 0.4])
            }
        }
        
        return trends
    
    def generate_content_strategy_report(self) -> Dict:
        """
        Generate comprehensive content strategy report.
        
        Returns:
            Dict: Detailed content strategy insights
        """
        # Get all post analyses
        df = self.get_all_posts_summary()
        if df.empty:
            return {"error": "No posts to analyze"}
        
        # Get individual post insights
        content_analyzer = self._get_content_analyzer()
        
        # Analyze top and bottom performers
        top_posts = df.nlargest(3, 'engagement_quality_score')
        bottom_posts = df.nsmallest(3, 'engagement_quality_score')
        
        # Collect insights from top performers
        top_performer_insights = []
        for _, post in top_posts.iterrows():
            metrics = content_analyzer.analyze_single_post(post['media_id'], self.name)
            if metrics:
                top_performer_insights.extend(metrics.actionable_insights)
        
        # Analyze patterns
        high_engagement_emotions = df[df['engagement_quality_score'] > 0.6]['top_emotion'].value_counts()
        viral_content_patterns = df[df['viral_score'] > 0.3]
        
        # Fix the correlation calculation - convert index to numeric values
        try:
            # Create a numeric sequence for correlation (post chronological order)
            df_reset = df.reset_index(drop=True)
            correlation = df_reset['engagement_quality_score'].corr(pd.Series(range(len(df_reset))))
            trend = "improving" if correlation > 0 else "declining"
        except Exception:
            # Fallback: compare first half vs second half performance
            mid_point = len(df) // 2
            if mid_point > 0:
                first_half_avg = df.iloc[:mid_point]['engagement_quality_score'].mean()
                second_half_avg = df.iloc[mid_point:]['engagement_quality_score'].mean()
                trend = "improving" if second_half_avg > first_half_avg else "declining"
            else:
                trend = "stable"
        
        # Determine key opportunity
        if viral_content_patterns.empty:
            key_opportunity = "viral_content_development"
        else:
            key_opportunity = f"{len(viral_content_patterns)} posts show viral potential - scale successful patterns"
        
        report = {
            "executive_summary": {
                "total_content_analyzed": len(df),
                "average_engagement_quality": round(df['engagement_quality_score'].mean(), 3),
                "content_performance_trend": trend,
                "key_opportunity": key_opportunity
            },
            "top_performing_content": {
                "posts": top_posts[['media_id', 'engagement_quality_score', 'top_emotion']].to_dict('records'),
                "common_characteristics": {
                    "dominant_emotions": high_engagement_emotions.head(3).to_dict() if not high_engagement_emotions.empty else {},
                    "avg_positive_sentiment": round(top_posts['positive_sentiment_pct'].mean(), 1),
                    "avg_viral_score": round(top_posts['viral_score'].mean(), 3)
                },
                "success_patterns": top_performer_insights[:5] if top_performer_insights else ["No specific insights available"]
            },
            "content_gaps": {
                "low_performing_posts": bottom_posts[['media_id', 'engagement_quality_score', 'negative_sentiment_pct']].to_dict('records'),
                "improvement_opportunities": [
                    f"Focus on {high_engagement_emotions.index[0]} emotion" if not high_engagement_emotions.empty else "Improve emotional engagement",
                    f"Reduce negative sentiment (avg: {bottom_posts['negative_sentiment_pct'].mean():.1f}%)" if 'negative_sentiment_pct' in bottom_posts.columns else "Monitor sentiment trends",
                    f"Increase viral potential (current avg: {df['viral_score'].mean():.3f})" if 'viral_score' in df.columns else "Develop sharing strategies"
                ]
            },
            "customer_insights": {
                "lifecycle_distribution": {
                    "discovery_focused_posts": len(df[df['loyalty_stage_comments'] < df['total_comments'] * 0.3]) if 'loyalty_stage_comments' in df.columns else 0,
                    "loyalty_focused_posts": len(df[df['loyalty_stage_comments'] > df['total_comments'] * 0.5]) if 'loyalty_stage_comments' in df.columns else 0
                },
                "engagement_patterns": {
                    "question_driving_posts": len(df[df['question_generation_rate'] > 20]) if 'question_generation_rate' in df.columns else 0,
                    "advice_seeking_posts": len(df[df['question_generation_rate'] > 15]) if 'question_generation_rate' in df.columns else 0,
                    "high_urgency_posts": len(df[df['high_urgency_comments'] > 2]) if 'high_urgency_comments' in df.columns else 0
                }
            },
            "actionable_recommendations": self._generate_content_recommendations(df),
            "performance_benchmarks": {
                "engagement_quality": {
                    "excellent": "> 0.7",
                    "good": "0.5 - 0.7", 
                    "needs_improvement": "< 0.5"
                },
                "current_distribution": {
                    "excellent": len(df[df['engagement_quality_score'] > 0.7]),
                    "good": len(df[(df['engagement_quality_score'] >= 0.5) & (df['engagement_quality_score'] <= 0.7)]),
                    "needs_improvement": len(df[df['engagement_quality_score'] < 0.5])
                }
            }
        }
        
        return report

    def _generate_content_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate specific content recommendations based on analysis."""
        recommendations = []
        
        # Engagement quality recommendations
        avg_quality = df['engagement_quality_score'].mean()
        if avg_quality < 0.5:
            recommendations.append("🎯 Focus on creating more engaging content - current avg quality is below benchmark")
        
        # Sentiment recommendations
        if 'positive_sentiment_pct' in df.columns:
            avg_positive = df['positive_sentiment_pct'].mean()
            if avg_positive < 60:
                recommendations.append(f"😊 Improve content positivity - current avg {avg_positive:.1f}% positive sentiment")
        
        # Viral potential recommendations
        if 'viral_score' in df.columns:
            avg_viral = df['viral_score'].mean()
            if avg_viral < 0.2:
                recommendations.append("📈 Increase shareability - add more user-tagging incentives and interactive elements")
        
        # Emotion-based recommendations
        if 'top_emotion' in df.columns:
            top_emotion = df['top_emotion'].mode().iloc[0] if not df['top_emotion'].mode().empty else None
            if top_emotion == 'confused':
                recommendations.append("💡 Create more educational content - high confusion indicates knowledge gaps")
            elif top_emotion == 'frustrated':
                recommendations.append("🚨 Address customer pain points - high frustration needs immediate attention")
            elif top_emotion == 'excited':
                recommendations.append("🎉 Leverage excitement - create user-generated content campaigns")
        
        # Question generation recommendations
        if 'question_generation_rate' in df.columns:
            avg_questions = df['question_generation_rate'].mean()
            if avg_questions > 25:
                recommendations.append("❓ High question generation - create FAQ content and educational series")
            elif avg_questions < 10:
                recommendations.append("🤔 Low question generation - make content more thought-provoking and interactive")
        
        # Urgency recommendations
        if 'high_urgency_comments' in df.columns:
            total_urgent = df['high_urgency_comments'].sum()
            if total_urgent > 20:
                recommendations.append(f"🚨 {total_urgent} high-urgency comments across posts - implement faster response protocols")
        
        # Performance variance recommendations
        quality_std = df['engagement_quality_score'].std()
        if quality_std > 0.3:
            recommendations.append("📊 High performance variance - analyze top performers and standardize successful elements")
        
        # Add default recommendations if none were generated
        if not recommendations:
            recommendations = [
                "📊 Continue monitoring engagement metrics to identify trends",
                "🎯 Focus on creating consistent, high-quality content",
                "📈 Experiment with different content formats to find what resonates"
            ]
        
        return recommendations[:7]  # Limit to top 7 recommendations
    
    def print_content_analysis_summary(self):
        """Print a comprehensive content analysis summary."""
        print("\n" + "=" * 80)
        print("📊 CONTENT PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Get content insights
        insights = self.get_content_insights_summary()
        
        if 'error' in insights:
            print("❌ No content data available for analysis")
            return
        
        print(f"📈 Content Overview:")
        print(f"   Total posts analyzed: {insights['total_posts_analyzed']}")
        print(f"   Average engagement quality: {insights['avg_engagement_quality']:.3f}")
        print(f"   Average positive sentiment: {insights['avg_positive_sentiment']:.1f}%")
        print(f"   Posts with viral potential: {insights['posts_with_high_viral_potential']}")
        print(f"   Most common emotion: {insights['most_common_top_emotion']}")
        print(f"   Posts driving questions: {insights['posts_driving_questions']}")
        
        # Get top performers
        top_posts = self.get_top_performing_posts(limit=5)
        
        if not top_posts.empty:
            print(f"\n🏆 Top 5 Performing Posts (by engagement quality):")
            for idx, post in top_posts.iterrows():
                print(f"   {post['media_id'][:15]}... - Quality: {post['engagement_quality_score']:.3f}, "
                      f"Emotion: {post['top_emotion']}, Comments: {post['total_comments']}")
        
        # Get trends
        trends = self.analyze_content_trends()
        
        print(f"\n📊 Performance Distribution:")
        perf_dist = trends['performance_distribution']
        print(f"   High quality (>0.7): {perf_dist['high_quality']} posts")
        print(f"   Medium quality (0.4-0.7): {perf_dist['medium_quality']} posts")
        print(f"   Low quality (<0.4): {perf_dist['low_quality']} posts")
        
        if trends['urgency_hotspots']:
            print(f"\n🚨 Posts with High-Urgency Comments:")
            for media_id in trends['urgency_hotspots'][:3]:  # Show top 3
                print(f"   {media_id[:20]}...")
        
        print(f"\n🎭 Emotion Patterns:")
        for emotion, count in list(trends['emotion_patterns'].items())[:5]:
            print(f"   {emotion.capitalize()}: {count} posts")

    def analyze_specific_post(self, media_id: str, detailed: bool = True):
        """
        Analyze and display detailed information about a specific post.
        
        Args:
            media_id (str): The media ID to analyze
            detailed (bool): Whether to show detailed insights
        """
        print(f"\n🔍 Analyzing Post: {media_id}")
        print("=" * 60)
        
        metrics = self.analyze_post_performance(media_id)
        
        if not metrics:
            print("❌ No data found for this post")
            return
        
        print(f"📊 Basic Metrics:")
        print(f"   Total comments: {metrics.total_comments}")
        print(f"   Engagement quality: {metrics.engagement_quality_score:.3f}")
        
        print(f"\n😊 Sentiment Distribution:")
        for sentiment, percentage in metrics.sentiment_distribution.items():
            print(f"   {sentiment.capitalize()}: {percentage:.1f}%")
        
        print(f"\n🎭 Top Emotions:")
        for emotion, score in metrics.top_emotions:
            print(f"   {emotion.capitalize()}: {score:.3f}")
        
        print(f"\n🚨 Urgency Breakdown:")
        for level, count in metrics.urgency_breakdown.items():
            if count > 0:
                print(f"   {level.capitalize()}: {count} comments")
        
        print(f"\n👥 Customer Lifecycle:")
        for stage, count in metrics.lifecycle_breakdown.items():
            if count > 0:
                print(f"   {stage.capitalize()}: {count} comments")
        
        print(f"\n📈 Viral Indicators:")
        viral = metrics.viral_indicators
        print(f"   User tagging rate: {viral['user_tagging_rate']:.1f}%")
        print(f"   Viral score: {viral['viral_score']:.3f}")
        
        print(f"\n💡 Content Effectiveness:")
        content_eff = metrics.content_effectiveness
        print(f"   Question generation: {content_eff['question_generation_rate']:.1f}%")
        print(f"   Advice seeking: {content_eff['advice_seeking_rate']:.1f}%")
        
        if detailed and metrics.actionable_insights:
            print(f"\n🎯 Actionable Insights:")
            for insight in metrics.actionable_insights:
                print(f"   {insight}")

    def export_content_analysis(self, output_path: str = "csv/") -> str:
        """
        Export comprehensive content analysis to CSV.
        
        Args:
            output_path (str): Optional path for output file
            
        Returns:
            str: Path to exported file
        """
        from datetime import datetime
        import os
    
        # Get comprehensive analysis
        df = self.get_all_posts_summary()
        
        if df.empty:
            print("❌ No data to export")
            return ""
        
        # Determine output path
        if output_path is None:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"📁 Created directory: {output_dir}")
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            company_clean = self.name.replace('@', '').replace(' ', '_')
            filename = f"content_analysis_{company_clean}_{timestamp}.csv"
            output_path = os.path.join(output_dir, filename)
        else:
            # Ensure the directory for the specified path exists
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Created directory: {directory}")
            
            # Add .csv extension if not present
            if not output_path.endswith('.csv'):
                output_path += '.csv'
        
        try:
            # Add detailed insights for each post
            detailed_data = []
            content_analyzer = self._get_content_analyzer()
            
            print(f"📊 Processing {len(df)} posts for detailed export...")
            
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                try:
                    metrics = content_analyzer.analyze_single_post(row['media_id'], self.name)
                    if metrics:
                        detailed_row = row.to_dict()
                        detailed_row.update({
                            'sentiment_negative_pct': metrics.sentiment_distribution.get('negative', 0),
                            'sentiment_neutral_pct': metrics.sentiment_distribution.get('neutral', 0),
                            'user_tagging_rate': metrics.viral_indicators.get('user_tagging_rate', 0),
                            'advice_seeking_rate': metrics.content_effectiveness.get('advice_seeking_rate', 0),
                            'urgency_high_count': metrics.urgency_breakdown.get('high', 0),
                            'urgency_medium_count': metrics.urgency_breakdown.get('medium', 0),
                            'lifecycle_discovery_count': metrics.lifecycle_breakdown.get('discovery', 0),
                            'lifecycle_consideration_count': metrics.lifecycle_breakdown.get('consideration', 0),
                            'top_insights': '; '.join(metrics.actionable_insights[:3]) if metrics.actionable_insights else '',
                            'emotion_excited': metrics.emotion_distribution.get('excited', 0),
                            'emotion_frustrated': metrics.emotion_distribution.get('frustrated', 0),
                            'emotion_grateful': metrics.emotion_distribution.get('grateful', 0),
                            'emotion_confused': metrics.emotion_distribution.get('confused', 0)
                        })
                        detailed_data.append(detailed_row)
                    else:
                        # Add basic row data even if detailed analysis fails
                        detailed_data.append(row.to_dict())
                    
                    # Progress indicator
                    if idx % 10 == 0 or idx == len(df):
                        print(f"   Processed {idx}/{len(df)} posts...")
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not get detailed analysis for post {row.get('media_id', 'unknown')}: {e}")
                    # Add basic row data if detailed analysis fails
                    detailed_data.append(row.to_dict())
                    continue
            
            # Create comprehensive DataFrame
            detailed_df = pd.DataFrame(detailed_data)
            
            # Export to CSV
            detailed_df.to_csv(output_path, index=False)
            
            print(f"✅ Content analysis exported successfully!")
            print(f"📄 File location: {output_path}")
            print(f"📊 Exported {len(detailed_df)} posts with {len(detailed_df.columns)} metrics")
            
            # Show sample of exported columns
            print(f"📋 Exported columns: {', '.join(detailed_df.columns[:10])}{'...' if len(detailed_df.columns) > 10 else ''}")
            
            return output_path
            
        except PermissionError as e:
            print(f"❌ Permission error: Cannot write to {output_path}")
            print(f"💡 Try running with administrator privileges or choose a different directory")
            print(f"📍 Current working directory: {os.getcwd()}")
            return ""
        except Exception as e:
            print(f"❌ Error exporting content analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

    def get_content_recommendations_dashboard(self) -> Dict:
        """
        Get a dashboard-ready summary of content recommendations.
        
        Returns:
            Dict: Dashboard data with key metrics and recommendations
        """
        # Get basic metrics
        insights = self.get_content_insights_summary()
        trends = self.analyze_content_trends()
        
        if 'error' in insights:
            return {"error": "No content data available"}
        
        # Get top and bottom performers
        all_posts = self.get_all_posts_summary()
        top_performer = all_posts.loc[all_posts['engagement_quality_score'].idxmax()]
        worst_performer = all_posts.loc[all_posts['engagement_quality_score'].idxmin()]
        
        dashboard = {
            "kpis": {
                "total_posts": insights['total_posts_analyzed'],
                "avg_engagement_quality": round(insights['avg_engagement_quality'], 3),
                "avg_positive_sentiment": round(insights['avg_positive_sentiment'], 1),
                "viral_potential_posts": insights['posts_with_high_viral_potential'],
                "high_urgency_total": insights['total_high_urgency_comments']
            },
            "performance_summary": {
                "best_performing_post": {
                    "media_id": top_performer['media_id'],
                    "quality_score": top_performer['engagement_quality_score'],
                    "dominant_emotion": top_performer['top_emotion']
                },
                "worst_performing_post": {
                    "media_id": worst_performer['media_id'], 
                    "quality_score": worst_performer['engagement_quality_score'],
                    "negative_sentiment": worst_performer['negative_sentiment_pct']
                },
                "performance_distribution": trends['performance_distribution']
            },
            "emotion_insights": {
                "most_common_emotion": insights['most_common_top_emotion'],
                "emotion_breakdown": trends['emotion_patterns']
            },
            "urgent_actions": {
                "posts_needing_attention": trends['urgency_hotspots'][:5],
                "total_urgent_comments": insights['total_high_urgency_comments']
            },
            "growth_opportunities": {
                "question_driving_posts": insights['posts_driving_questions'],
                "viral_potential_untapped": insights['total_posts_analyzed'] - insights['posts_with_high_viral_potential']
            },
            "recommendations": self._generate_content_recommendations(all_posts)[:5]
        }
        
        return dashboard

    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def create_database_and_tables(self):
        """Create all necessary database tables"""
        conn = self._ensure_connection()
        cursor = conn.cursor()

        # Create company table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS company (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                business_type TEXT NOT NULL
            )
        ''')
        
        # Create engagement table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engagement (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                media_id TEXT,
                media_caption TEXT,
                comment_text TEXT,
                company_id INTEGER,
                FOREIGN KEY (company_id) REFERENCES company (id)
            )
        ''')
        
        # Create comment_analysis table with transformer sentiment fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engagement_id INTEGER NOT NULL,
                word_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                has_question BOOLEAN DEFAULT 0,
                has_advice_seeking BOOLEAN DEFAULT 0,
                emotion_indicators TEXT,  -- JSON: {"excited": 0.8, "frustrated": 0.2}
                urgency_level TEXT DEFAULT 'low',  -- low/medium/high
                customer_lifecycle_stage TEXT DEFAULT 'discovery',  -- discovery/consideration/loyalty  
                tags_users BOOLEAN DEFAULT 0,
                tagged_usernames TEXT,  -- JSON array of @usernames
                sentiment_negative REAL DEFAULT 0.0,  -- Transformer negative score
                sentiment_neutral REAL DEFAULT 0.0,   -- Transformer neutral score
                sentiment_positive REAL DEFAULT 0.0,  -- Transformer positive score
                sentiment_label TEXT DEFAULT 'neutral',  -- negative/neutral/positive
                sentiment_confidence REAL DEFAULT 0.0,  -- Confidence in prediction
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engagement_id) REFERENCES engagement (id),
                UNIQUE(engagement_id)
            )
        ''')
        
        # Add indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comment_analysis_engagement ON comment_analysis(engagement_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comment_analysis_sentiment ON comment_analysis(sentiment_label)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comment_analysis_urgency ON comment_analysis(urgency_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comment_analysis_lifecycle ON comment_analysis(customer_lifecycle_stage)')
        
        # Commit the table creation
        conn.commit()
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float, float, str, float]:
        """
        Analyze sentiment using the specialized SentimentAnalyzer.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            Tuple[float, float, float, str, float]: 
                (negative_score, neutral_score, positive_score, label, confidence)
        """
        if not self.sentiment_analyzer:
            return 0.0, 1.0, 0.0, 'neutral', 0.5
        
        return self.sentiment_analyzer.analyze_sentiment_transformers(text)
    
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions using the specialized EmotionAnalyzer.
        
        Args:
            text (str): Comment text to analyze
            
        Returns:
            Dict[str, float]: Emotion scores (0.0 to 1.0)
        """
        if not self.emotion_analyzer:
            return {
                "excited": 0.0, "frustrated": 0.0, "grateful": 0.0, "confused": 0.0,
                "satisfied": 0.0, "disappointed": 0.0, "hopeful": 0.0, "desperate": 0.0
            }
        
        return self.emotion_analyzer.analyze_emotions(text)
    
    def determine_urgency_level(self, text: str, emotions: Dict[str, float]) -> str:
        """
        Determine urgency level with skincare context.
        
        Args:
            text (str): Comment text
            emotions (Dict[str, float]): Emotion scores from emotion analyzer
            
        Returns:
            str: 'low', 'medium', or 'high'
        """
        text_lower = text.lower()
        
        # High urgency indicators (skincare-specific)
        high_urgency_patterns = [
            'asap', 'urgent', 'emergency', 'desperate', 'please help', 'getting worse',
            'severe', 'painful', 'cant take it', "can't take it", 'breaking out badly',
            'wedding', 'event', 'tomorrow', 'today', 'right now', 'immediately',
            'cystic acne', 'severe breakout', 'allergic reaction', 'burning sensation'
        ]
        
        # Medium urgency indicators  
        medium_urgency_patterns = [
            'soon', 'quickly', 'fast', 'need help', 'advice needed', 'what should i do',
            'getting bad', 'worried', 'concern', 'spreading', 'worsening',
            'running out', 'need new routine', 'not working anymore'
        ]
        
        # Emotional urgency factors - Use threshold values
        emotional_urgency = (
            emotions.get("desperate", 0.0) > 0.5 or 
            emotions.get("frustrated", 0.0) > 0.7
        )
        
        # Check for multiple exclamation marks or caps (urgency indicators)
        has_multiple_exclamation = len(re.findall(r'!{2,}', text)) > 0
        has_excessive_caps = len(re.findall(r'[A-Z]{3,}', text)) > 0
        
        if any(pattern in text_lower for pattern in high_urgency_patterns) or emotional_urgency:
            return 'high'
        elif (any(pattern in text_lower for pattern in medium_urgency_patterns) or 
              has_multiple_exclamation or has_excessive_caps):
            return 'medium'
        else:
            return 'low'
    
    def determine_customer_lifecycle_stage(self, text: str, emotions: Dict[str, float]) -> str:
        """
        Determine customer lifecycle stage with skincare context.
        
        Args:
            text (str): Comment text
            emotions (Dict[str, float]): Emotion scores from emotion analyzer
            
        Returns:
            str: 'discovery', 'consideration', or 'loyalty'
        """
        text_lower = text.lower()
        
        # Discovery stage indicators
        discovery_patterns = [
            'heard about', 'saw this', 'first time', 'just discovered', 'new to', 
            'what is this', 'never tried', 'thinking about', 'considering',
            'influencer recommended', 'saw on tiktok', 'instagram ad'
        ]
        
        # Consideration stage indicators
        consideration_patterns = [
            'vs', 'compare', 'which one', 'better than', 'should i buy', 
            'worth it', 'reviews', 'anyone tried', 'how does this work',
            'ingredients', 'before and after', 'results', 'experiences'
        ]
        
        # Loyalty stage indicators
        loyalty_patterns = [
            'repurchase', 'buying again', 'love this', 'always use', 'go-to',
            'holy grail', 'staple', 'been using', 'monthly order', 'subscription',
            'recommend', 'told my friends', 'third bottle', 'reordering'
        ]
        
        # Use threshold to determine strong gratitude
        if (any(pattern in text_lower for pattern in loyalty_patterns) or 
            emotions.get("grateful", 0.0) > 0.5):
            return 'loyalty'
        elif any(pattern in text_lower for pattern in consideration_patterns):
            return 'consideration'
        else:
            return 'discovery'
    
    def extract_tagged_users(self, text: str) -> Tuple[bool, List[str]]:
        """Extract Instagram user tags from comment text"""
        tag_pattern = r'@([a-zA-Z0-9_.]+)'
        tagged_users = re.findall(tag_pattern, text)
        
        # Filter out common false positives
        filtered_users = []
        for user in tagged_users:
            if user.lower() not in ['treehut', 'skincare', 'beauty'] and len(user) > 2:
                filtered_users.append(user)
        
        return len(filtered_users) > 0, filtered_users
    
    def analyze_single_comment(self, comment_text: str) -> CommentMetrics:
        """Comprehensive analysis of a single comment using specialized analyzers"""
        if not comment_text or comment_text.strip() == '':
            return CommentMetrics(
                word_count=0, char_count=0, has_question=False, has_advice_seeking=False,
                emotion_indicators='{}', urgency_level='low', customer_lifecycle_stage='discovery',
                tags_users=False, tagged_usernames='[]', 
                sentiment_negative=0.0, sentiment_neutral=1.0, sentiment_positive=0.0,
                sentiment_label='neutral', sentiment_confidence=0.5
            )
        
        # Basic text metrics
        word_count = len(comment_text.split())
        char_count = len(comment_text)
        
        # Question detection
        has_question = '?' in comment_text
        
        # Advice-seeking detection
        advice_patterns = ['help', 'advice', 'recommend', 'suggest', 'what should', 'how do', 'which']
        has_advice_seeking = any(pattern in comment_text.lower() for pattern in advice_patterns)
        
        # Emotional analysis using specialized analyzer
        emotions = self.analyze_emotions(comment_text)
        
        # Urgency analysis
        urgency_level = self.determine_urgency_level(comment_text, emotions)
        
        # Lifecycle stage
        lifecycle_stage = self.determine_customer_lifecycle_stage(comment_text, emotions)
        
        # User tagging analysis
        tags_users, tagged_usernames = self.extract_tagged_users(comment_text)
        
        # Sentiment analysis using specialized analyzer
        neg_score, neu_score, pos_score, sentiment_label, confidence = self.analyze_sentiment(comment_text)
        
        return CommentMetrics(
            word_count=word_count,
            char_count=char_count,
            has_question=has_question,
            has_advice_seeking=has_advice_seeking,
            emotion_indicators=json.dumps(emotions),  # Stores float scores as JSON
            urgency_level=urgency_level,
            customer_lifecycle_stage=lifecycle_stage,
            tags_users=tags_users,
            tagged_usernames=json.dumps(tagged_usernames),
            sentiment_negative=neg_score,
            sentiment_neutral=neu_score,
            sentiment_positive=pos_score,
            sentiment_label=sentiment_label,
            sentiment_confidence=confidence
        )
    
    def populate_comment_analysis(self, company_name: str = None, batch_size: int = 50) -> bool:
        """
        Analyze all comments and populate the comment_analysis table
        Uses specialized analyzers for processing
        """
        if company_name is None:
            company_name = self.name
            
        # Check if analyzers are available
        if not self.sentiment_analyzer or not self.emotion_analyzer:
            print("❌ Analyzers not available. Please install transformers.")
            return False
        
        # Check if sentiment analyzer has transformers loaded
        if not self.sentiment_analyzer.sentiment_pipeline:
            print("❌ Transformer models not available. Please install transformers.")
            return False
            
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        print(f"🔍 Analyzing comments for {company_name} using specialized analyzers...")
        
        # Get all engagements for the company
        cursor.execute('''
            SELECT e.id, e.comment_text
            FROM engagement e
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ? AND e.comment_text IS NOT NULL AND e.comment_text != ''
        ''', (company_name,))
        
        engagements = cursor.fetchall()
        
        if not engagements:
            print(f"⚠️  No engagements found for {company_name}")
            return False
        
        print(f"📊 Processing {len(engagements)} comments with specialized analysis...")
        print(f"⏱️  Estimated time: {len(engagements) * 0.1 / 60:.1f} minutes")
        
        processed_count = 0
        start_time = time.time()
        
        for i in range(0, len(engagements), batch_size):
            batch = engagements[i:i + batch_size]
            batch_start = time.time()
            
            for engagement_id, comment_text in batch:
                try:
                    # Analyze the comment using specialized analyzers
                    metrics = self.analyze_single_comment(comment_text)
                    
                    # Insert analysis results
                    cursor.execute('''
                        INSERT OR REPLACE INTO comment_analysis 
                        (engagement_id, word_count, char_count, has_question, has_advice_seeking,
                         emotion_indicators, urgency_level, customer_lifecycle_stage,
                         tags_users, tagged_usernames, sentiment_negative, sentiment_neutral,
                         sentiment_positive, sentiment_label, sentiment_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        engagement_id, metrics.word_count, metrics.char_count,
                        metrics.has_question, metrics.has_advice_seeking,
                        metrics.emotion_indicators, metrics.urgency_level,
                        metrics.customer_lifecycle_stage, metrics.tags_users,
                        metrics.tagged_usernames, metrics.sentiment_negative,
                        metrics.sentiment_neutral, metrics.sentiment_positive,
                        metrics.sentiment_label, metrics.sentiment_confidence
                    ))
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing comment {engagement_id}: {e}")
                    continue
            
            # Commit batch
            conn.commit()
            
            batch_time = time.time() - batch_start
            print(f"   Processed batch {i//batch_size + 1}/{(len(engagements) + batch_size - 1)//batch_size} "
                  f"({processed_count}/{len(engagements)}) - {batch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Analysis complete! Processed {processed_count} comments in {total_time:.1f} seconds.")
        
        # Show summary statistics
        self._show_analysis_summary(company_name)
        
        return True
    
    def _show_analysis_summary(self, company_name: str):
        """Show detailed analysis summary"""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_analyzed,
                SUM(has_question) as questions,
                SUM(has_advice_seeking) as advice_seeking,
                SUM(tags_users) as user_tags,
                AVG(sentiment_confidence) as avg_confidence
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        ''', (company_name,))
        
        stats = cursor.fetchone()
        total_comments = stats[0]
        
        if total_comments == 0:
            print("⚠️  No analyzed comments found.")
            return
        
        print(f"\n📈 Analysis Summary:")
        print(f"   Total comments: {total_comments}")
        print(f"   Questions: {stats[1]} ({stats[1]/total_comments*100:.1f}%)")
        print(f"   Advice-seeking: {stats[2]} ({stats[2]/total_comments*100:.1f}%)")
        print(f"   User tags: {stats[3]} ({stats[3]/total_comments*100:.1f}%)")
        print(f"   Avg confidence: {stats[4]:.3f}")
        
        # Sentiment breakdown
        cursor.execute('''
            SELECT sentiment_label, COUNT(*) as count
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY sentiment_label
        ''', (company_name,))
        
        sentiment_stats = cursor.fetchall()
        print(f"\n😊 Sentiment Breakdown:")
        for label, count in sentiment_stats:
            print(f"   {label.capitalize()}: {count} ({count/total_comments*100:.1f}%)")
        
        # Urgency breakdown
        cursor.execute('''
            SELECT urgency_level, COUNT(*) as count
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY urgency_level
        ''', (company_name,))
        
        urgency_stats = cursor.fetchall()
        print(f"\n🚨 Urgency Breakdown:")
        for level, count in urgency_stats:
            print(f"   {level.capitalize()}: {count} ({count/total_comments*100:.1f}%)")
        
        # Show top emotions detected
        cursor.execute('''
            SELECT emotion_indicators
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ? AND emotion_indicators != '{}'
            LIMIT 100
        ''', (company_name,))
        
        emotion_data = cursor.fetchall()
        if emotion_data:
            # Aggregate emotion scores
            emotion_totals = {
                "excited": 0.0, "frustrated": 0.0, "grateful": 0.0, "confused": 0.0,
                "satisfied": 0.0, "disappointed": 0.0, "hopeful": 0.0, "desperate": 0.0
            }
            
            for (emotion_json,) in emotion_data:
                try:
                    emotions = json.loads(emotion_json)
                    for emotion, score in emotions.items():
                        if emotion in emotion_totals:
                            emotion_totals[emotion] += score
                except:
                    continue
            
            print(f"\n🎭 Top Emotions Detected:")
            sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
            for emotion, total_score in sorted_emotions[:5]:
                avg_score = total_score / len(emotion_data)
                if avg_score > 0.01:  # Only show emotions with meaningful scores
                    print(f"   {emotion.capitalize()}: {avg_score:.3f} avg score")
    
    def insert_company_data(self) -> int:
        """Insert company data and return company_id"""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO company (name, business_type)
            VALUES (?, ?)
        ''', (self.name, self.type))
        
        cursor.execute('SELECT id FROM company WHERE name = ?', (self.name,))
        result = cursor.fetchone()
        
        if result is None:
            raise ValueError(f"Failed to insert or find company: {self.name}")
        
        company_id = result[0]
        conn.commit()
        return company_id

    def load_csv_to_database(self, csv_file_path: str, overwrite=True) -> bool:
        """Main function to load CSV data into SQLite database"""
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file '{csv_file_path}' not found!")
            return False
        
        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()
            
            print("Creating database and tables...")
            self.create_database_and_tables()
            
            print("Inserting company data...")
            company_id = self.insert_company_data()
            
            print(f"Reading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            # Add company_id column
            df['company_id'] = company_id
            
            # Handle overwrite by clearing existing data, NOT replacing table structure
            if overwrite:
                print("🗑️  Clearing existing engagement data...")
                cursor.execute('DELETE FROM engagement WHERE company_id = ?', (company_id,))
                cursor.execute('DELETE FROM comment_analysis WHERE engagement_id NOT IN (SELECT id FROM engagement)')
                conn.commit()
                print("✅ Existing data cleared")
            
            print("Inserting engagement data...")
            # Always use 'append' to preserve table structure
            df.to_sql('engagement', conn, if_exists='append', index=False)
            
            # Verify insertion
            cursor.execute('SELECT COUNT(*) FROM engagement WHERE company_id = ?', (company_id,))
            record_count = cursor.fetchone()[0]
            
            print(f"Successfully inserted {record_count} engagement records!")
            
            # Show sample data to verify structure
            cursor.execute('''
                SELECT e.id, e.timestamp, e.media_id, e.comment_text
                FROM engagement e
                WHERE e.company_id = ?
                LIMIT 3
            ''', (company_id,))
            
            sample_rows = cursor.fetchall()
            print("\n📋 Sample data:")
            for row in sample_rows:
                comment_preview = row[3][:50] + "..." if row[3] and len(row[3]) > 50 else row[3]
                print(f"   ID: {row[0]}, Timestamp: {row[1]}, Media: {row[2][:10]}...")
                print(f"   Comment: {comment_preview}")
            
            print(f"Database saved as: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()  # Show full error details
            return False
        
    # =====================================
    # VISUALIZATION METHODS
    # =====================================
    def create_sentiment_overview_dashboard(self) -> go.Figure:
        """Create comprehensive sentiment analysis dashboard"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")
        
        # Get sentiment data
        sentiment_data = self._get_sentiment_data()
        
        if sentiment_data.empty:
            return self._create_no_data_figure("No sentiment data available")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment Distribution', 
                'Sentiment Confidence Over Time',
                'Sentiment by Customer Lifecycle Stage',
                'High-Confidence Sentiment Breakdown'
            ),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = sentiment_data['sentiment_label'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name="Sentiment Distribution",
                marker_colors=[self.colors['sentiment'][label] for label in sentiment_counts.index],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Sentiment Confidence Over Time
        daily_sentiment = self._get_daily_sentiment_data()
        
        if not daily_sentiment.empty:
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_sentiment.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sentiment.index,
                            y=daily_sentiment[sentiment],
                            mode='lines+markers',
                            name=f'{sentiment.capitalize()}',
                            line=dict(color=self.colors['sentiment'][sentiment]),
                            hovertemplate=f'<b>{sentiment.capitalize()}</b><br>Date: %{{x}}<br>Avg Score: %{{y:.3f}}<extra></extra>'
                        ),
                        row=1, col=2
                    )
        
        # 3. Sentiment by Lifecycle Stage
        lifecycle_sentiment = self._get_lifecycle_sentiment_data()
        
        if not lifecycle_sentiment.empty:
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in lifecycle_sentiment.columns:
                    fig.add_trace(
                        go.Bar(
                            x=lifecycle_sentiment.index,
                            y=lifecycle_sentiment[sentiment],
                            name=f'{sentiment.capitalize()}',
                            marker_color=self.colors['sentiment'][sentiment],
                            showlegend=False,
                            hovertemplate=f'<b>{sentiment.capitalize()}</b><br>Stage: %{{x}}<br>Count: %{{y}}<extra></extra>'
                        ),
                        row=2, col=1
                    )
        
        # 4. High-Confidence Sentiment
        high_conf_data = sentiment_data[sentiment_data['sentiment_confidence'] > 0.8]
        if not high_conf_data.empty:
            high_conf_counts = high_conf_data['sentiment_label'].value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=high_conf_counts.index,
                    y=high_conf_counts.values,
                    name="High Confidence",
                    marker_color=[self.colors['sentiment'][label] for label in high_conf_counts.index],
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>High-Confidence Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Sentiment Analysis Dashboard - {self.name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_emotion_heatmap(self) -> go.Figure:
        """Create emotion intensity heatmap"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations")
        
        emotion_data = self._get_emotion_heatmap_data()
        
        if emotion_data.empty:
            return self._create_no_data_figure("No emotion data available")
        
        # Prepare heatmap data
        emotions = ['excited', 'frustrated', 'grateful', 'confused', 'satisfied', 'disappointed', 'hopeful', 'desperate']
        
        # Filter emotions to only include those that exist in the data
        available_emotions = [emotion for emotion in emotions if emotion in emotion_data.columns]
        
        if not available_emotions:
            return self._create_no_data_figure("No emotion data available in the dataset")
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=emotion_data[available_emotions].values,
            x=available_emotions,
            y=emotion_data.index.strftime('%Y-%m-%d') if hasattr(emotion_data.index, 'strftime') else emotion_data.index,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{x}</b><br>Period: %{y}<br>Intensity: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title="Emotion Intensity",
                title_side="right"  # Fixed: changed from 'titleside' to 'title_side'
            )
        ))
        
        fig.update_layout(
            title=f'Emotion Intensity Heatmap - {self.name}',
            xaxis_title='Emotions',
            yaxis_title='Time Period',
            height=600,
            template='plotly_white',
            xaxis=dict(tickangle=45)  # Rotate emotion labels for better readability
        )
        
        return fig
    
    def create_urgency_dashboard(self) -> go.Figure:
        """Create urgency and priority management dashboard"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations")
        
        urgency_data = self._get_urgency_data()
        
        if urgency_data.empty:
            return self._create_no_data_figure("No urgency data available")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Urgency Distribution Over Time',
                'High-Urgency Comments by Sentiment',
                'Urgency vs. Word Count',
                'Priority Matrix (Urgency vs Negative Sentiment)'
            ),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Urgency Over Time
        daily_urgency = self._get_daily_urgency_data()
        
        if not daily_urgency.empty:
            for urgency in ['high', 'medium', 'low']:
                if urgency in daily_urgency.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_urgency.index,
                            y=daily_urgency[urgency],
                            mode='lines+markers',
                            name=f'{urgency.capitalize()} Urgency',
                            line=dict(color=self.colors['urgency'][urgency]),
                            hovertemplate=f'<b>{urgency.capitalize()} Urgency</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # 2. High-Urgency by Sentiment
        high_urgency = urgency_data[urgency_data['urgency_level'] == 'high']
        if not high_urgency.empty:
            sentiment_counts = high_urgency['sentiment_label'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="High-Urgency Sentiment",
                    marker_colors=[self.colors['sentiment'][label] for label in sentiment_counts.index],
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>High-Urgency Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Urgency vs Word Count
        for urgency in ['high', 'medium', 'low']:
            urgency_subset = urgency_data[urgency_data['urgency_level'] == urgency]
            if not urgency_subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=urgency_subset['word_count'],
                        y=urgency_subset['sentiment_confidence'],
                        mode='markers',
                        name=f'{urgency.capitalize()} Urgency',
                        marker=dict(
                            color=self.colors['urgency'][urgency],
                            size=8,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{urgency.capitalize()} Urgency</b><br>Word Count: %{{x}}<br>Sentiment Confidence: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Priority Matrix
        urgency_map = {'low': 1, 'medium': 2, 'high': 3}
        urgency_data['urgency_numeric'] = urgency_data['urgency_level'].map(urgency_map)
        urgency_data['priority_score'] = urgency_data['urgency_numeric'] * urgency_data['sentiment_negative']
        
        fig.add_trace(
            go.Scatter(
                x=urgency_data['urgency_numeric'],
                y=urgency_data['sentiment_negative'],
                mode='markers',
                marker=dict(
                    size=urgency_data['priority_score'] * 20,
                    color=urgency_data['priority_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Priority Score", x=1.02),
                    line=dict(width=1, color='white')
                ),
                name='Priority Matrix',
                hovertemplate='<b>Priority Analysis</b><br>Urgency Level: %{x}<br>Negative Sentiment: %{y:.3f}<br>Priority Score: %{marker.color:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Urgency & Priority Dashboard - {self.name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_comprehensive_dashboard(self) -> go.Figure:
        """Create a comprehensive dashboard with key metrics"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations")
        
        # Get data for various metrics
        sentiment_data = self._get_sentiment_data()
        emotion_data = self._get_emotion_summary_data()
        urgency_data = self._get_urgency_data()
        lifecycle_data = self._get_lifecycle_data()
        
        if sentiment_data.empty:
            return self._create_no_data_figure("No data available for dashboard")
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Sentiment Distribution',
                'Top 5 Emotions Detected', 
                'Urgency Levels',
                'Customer Lifecycle Stages',
                'Sentiment Confidence Distribution',
                'Comments with Questions vs Advice-Seeking'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment Distribution
        sentiment_counts = sentiment_data['sentiment_label'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[self.colors['sentiment'][label] for label in sentiment_counts.index],
                name="Sentiment"
            ),
            row=1, col=1
        )
        
        # 2. Top Emotions
        if not emotion_data.empty:
            top_emotions = emotion_data.head(5)
            fig.add_trace(
                go.Bar(
                    x=top_emotions.index,
                    y=top_emotions.values,
                    marker_color=[self.colors['emotions'].get(emotion, self.colors['primary']) 
                                for emotion in top_emotions.index],
                    name="Top Emotions"
                ),
                row=1, col=2
            )
        
        # 3. Urgency Levels
        if not urgency_data.empty:
            urgency_counts = urgency_data['urgency_level'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=urgency_counts.index,
                    values=urgency_counts.values,
                    marker_colors=[self.colors['urgency'][level] for level in urgency_counts.index],
                    name="Urgency"
                ),
                row=2, col=1
            )
        
        # 4. Lifecycle Stages
        if not lifecycle_data.empty:
            lifecycle_counts = lifecycle_data['customer_lifecycle_stage'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=lifecycle_counts.index,
                    values=lifecycle_counts.values,
                    marker_colors=[self.colors['lifecycle'][stage] for stage in lifecycle_counts.index],
                    name="Lifecycle"
                ),
                row=2, col=2
            )
        
        # 5. Sentiment Confidence Distribution
        fig.add_trace(
            go.Histogram(
                x=sentiment_data['sentiment_confidence'],
                nbinsx=20,
                marker_color=self.colors['info'],
                name="Confidence Distribution"
            ),
            row=3, col=1
        )
        
        # 6. Questions vs Advice-Seeking
        questions_count = sentiment_data['has_question'].sum()
        advice_count = sentiment_data['has_advice_seeking'].sum()
        
        fig.add_trace(
            go.Bar(
                x=['Questions', 'Advice-Seeking'],
                y=[questions_count, advice_count],
                marker_color=[self.colors['primary'], self.colors['secondary']],
                name="Content Types"
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title=f'Comprehensive Analytics Dashboard - {self.name}',
            height=1000,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def save_dashboard_html(self, fig: go.Figure, filename: str, auto_open: bool = False) -> str:
        """Save dashboard as interactive HTML file"""
        if not filename.endswith('.html'):
            filename += '.html'
        
        fig.write_html(filename, auto_open=auto_open)
        print(f"📊 Dashboard saved as: {filename}")
        return filename
    
    def _create_no_data_figure(self, message: str) -> go.Figure:
        """Create a figure showing no data message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            template='plotly_white'
        )
        return fig

    # =====================================
    # DATA RETRIEVAL METHODS FOR VISUALIZATIONS
    # =====================================
    
    def _get_sentiment_data(self) -> pd.DataFrame:
        """Get sentiment analysis data"""
        query = '''
            SELECT ca.sentiment_label, ca.sentiment_confidence, ca.sentiment_positive,
                   ca.sentiment_negative, ca.sentiment_neutral, ca.customer_lifecycle_stage,
                   ca.has_question, ca.has_advice_seeking, ca.urgency_level, ca.word_count,
                   e.timestamp
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        '''
        return pd.read_sql_query(query, self.connection, params=(self.name,))
    
    def _get_daily_sentiment_data(self) -> pd.DataFrame:
        """Get daily aggregated sentiment data"""
        query = '''
            SELECT DATE(e.timestamp) as date,
                   AVG(ca.sentiment_positive) as positive,
                   AVG(ca.sentiment_negative) as negative,
                   AVG(ca.sentiment_neutral) as neutral
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY DATE(e.timestamp)
            ORDER BY date
        '''
        df = pd.read_sql_query(query, self.connection, params=(self.name,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def _get_lifecycle_sentiment_data(self) -> pd.DataFrame:
        """Get sentiment data by customer lifecycle stage"""
        query = '''
            SELECT ca.customer_lifecycle_stage,
                   SUM(CASE WHEN ca.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN ca.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative,
                   SUM(CASE WHEN ca.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY ca.customer_lifecycle_stage
        '''
        df = pd.read_sql_query(query, self.connection, params=(self.name,))
        if not df.empty:
            df.set_index('customer_lifecycle_stage', inplace=True)
        return df
    
    def _get_emotion_heatmap_data(self) -> pd.DataFrame:
        """Get emotion data for heatmap visualization"""
        query = '''
            SELECT DATE(e.timestamp) as date, ca.emotion_indicators
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ? AND ca.emotion_indicators != '{}'
        '''
        df = pd.read_sql_query(query, self.connection, params=(self.name,))
        
        if df.empty:
            return df
        
        # Parse emotion indicators and aggregate by date
        emotions = ['excited', 'frustrated', 'grateful', 'confused', 'satisfied', 'disappointed', 'hopeful', 'desperate']
        emotion_data = []
        
        for _, row in df.iterrows():
            try:
                emotion_dict = json.loads(row['emotion_indicators'])
                emotion_dict['date'] = row['date']
                emotion_data.append(emotion_dict)
            except:
                continue
        
        if not emotion_data:
            return pd.DataFrame()
        
        emotion_df = pd.DataFrame(emotion_data)
        emotion_df['date'] = pd.to_datetime(emotion_df['date'])
        
        # Group by date and calculate mean emotions
        emotion_summary = emotion_df.groupby('date')[emotions].mean()
        
        return emotion_summary
    
    def _get_emotion_summary_data(self) -> pd.Series:
        """Get aggregated emotion data"""
        query = '''
            SELECT ca.emotion_indicators
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ? AND ca.emotion_indicators != '{}'
        '''
        df = pd.read_sql_query(query, self.connection, params=(self.name,))
        
        if df.empty:
            return pd.Series()
        
        # Aggregate all emotions
        emotion_totals = {
            "excited": 0.0, "frustrated": 0.0, "grateful": 0.0, "confused": 0.0,
            "satisfied": 0.0, "disappointed": 0.0, "hopeful": 0.0, "desperate": 0.0
        }
        
        count = 0
        for _, row in df.iterrows():
            try:
                emotions = json.loads(row['emotion_indicators'])
                for emotion, score in emotions.items():
                    if emotion in emotion_totals:
                        emotion_totals[emotion] += score
                count += 1
            except:
                continue
        
        # Calculate averages
        if count > 0:
            for emotion in emotion_totals:
                emotion_totals[emotion] /= count
        
        return pd.Series(emotion_totals).sort_values(ascending=False)
    
    def _get_urgency_data(self) -> pd.DataFrame:
        """Get urgency analysis data"""
        query = '''
            SELECT ca.urgency_level, ca.sentiment_label, ca.word_count, 
                   ca.sentiment_confidence, ca.sentiment_negative, ca.sentiment_positive
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        '''
        return pd.read_sql_query(query, self.connection, params=(self.name,))
    
    def _get_daily_urgency_data(self) -> pd.DataFrame:
        """Get daily urgency data"""
        query = '''
            SELECT DATE(e.timestamp) as date,
                   SUM(CASE WHEN ca.urgency_level = 'high' THEN 1 ELSE 0 END) as high,
                   SUM(CASE WHEN ca.urgency_level = 'medium' THEN 1 ELSE 0 END) as medium,
                   SUM(CASE WHEN ca.urgency_level = 'low' THEN 1 ELSE 0 END) as low
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY DATE(e.timestamp)
            ORDER BY date
        '''
        df = pd.read_sql_query(query, self.connection, params=(self.name,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def _get_lifecycle_data(self) -> pd.DataFrame:
        """Get customer lifecycle data"""
        query = '''
            SELECT ca.customer_lifecycle_stage, ca.sentiment_label
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        '''
        return pd.read_sql_query(query, self.connection, params=(self.name,))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()