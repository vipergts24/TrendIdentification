
import sqlite3
import pandas as pd
import os
import re
import json
import time
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from models.emotion_analyzer import EmotionAnalyzer
from models.sentiment_analyzer  import SentimentAnalyzer

from models.company_engagement import CommentMetrics,CompanyEngagement
from models.content_analyzer import ContentAnalyzer, ContentMetrics


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    CSV_FILE = "csv/engagements.csv"
    DATABASE_FILE = "data/engagements.db"
    
    # WARNING: sentiment analysis can take upwards of 10 min to run 
    # so may want to skip unless you want to populate full data
    SKIP_COMMENT_ANALYSIS = True
    
    # Run the script
    with CompanyEngagement("@treehut", "skincare", db_path=DATABASE_FILE) as company:
        # Load initial data
        success = company.load_csv_to_database(CSV_FILE, overwrite=True)
        
        if success:
            print("\n🤖 Starting transformer-based comment analysis...")
            # Run comment analysis with transformers
            if not SKIP_COMMENT_ANALYSIS:
                analysis_success = company.populate_comment_analysis(batch_size=64)
            
                if analysis_success:
                    print("🎉 Complete transformer analysis pipeline finished!")
                    print("\n💡 Next steps:")
                    print("   - Query sentiment trends over time")
                    print("   - Analyze high-confidence negative comments")
                    print("   - Identify positive sentiment drivers")
                    print("   - Correlate sentiment with engagement quality")
            else:
                print("⏭️  Skipping transformer analysis")
        else:
            print("❌ Failed to load CSV data")