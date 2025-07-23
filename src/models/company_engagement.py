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
        
        # Initialize our specialized analyzers
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        
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
        """Initialize the specialized sentiment and emotion analyzers"""
        print("🧠 Initializing specialized analyzers...")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize emotion analyzer
        self.emotion_analyzer = EmotionAnalyzer()
        
        print("✅ All analyzers initialized successfully!")

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
        
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()