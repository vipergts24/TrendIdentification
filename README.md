#  Treehut Trend Identification

Brief description of what your project does and why it exists.

## Setup

### Prerequisites
- Python 3.11+ 
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vipergts24/TrendIdentification.git
cd TrendIdentification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start
```python
from models.company_engagement import CompanyEngagement

# Initialize with your data
with CompanyEngagement("@yourcompany", "industry") as company:
    # Load your engagement data
    company.load_csv_to_database("your_data.csv")
    
    # Run sentiment/emotion analysis (will take considerable time for large datasets)
    company.populate_comment_analysis()
    
    # Generate all visualizations
    company.create_sentiment_overview_dashboard()
    company.create_emotion_heatmap()
    company.create_urgency_dashboard()
    company.create_comprehensive_dashboard()
```


## Features
- Sentiment Analysis on comments using Huggingface transformers
- Emotion Analysis on comments using Huggingface transformers
- Content Analysis on posts
- Visualizations to analyze metrics we've calculated


# Executive Summary

I built this project in mind to dive deep into into social media engagement data and create something that could actually simulate how a media manager can extract interesting metrics from pure textual comments.  

The goal was to create a complete analytics pipeline that could take thousands of social media comments and turn them into actionable intelligence for content strategy and customer engagement and could easily be extended to additional companies with their own engagement data.

## What I Built

### Database & Data Pipeline
I set up a SQLite database to handle all the engagement data efficiently. The schema supports complex queries across companies, posts, and individual comment analysis while keeping everything organized and fast.  It supports multiple companies but the analysis is primarily focused on skincare (due to time constraint).

### Core Analysis Components

**Sentiment Analysis**
- Used Hugging Face transformers to automatically score each comment as positive, negative, or neutral
- Added confidence scoring so you know when the model is uncertain
- Processes comments in batches to handle large datasets without breaking

**Emotion Detection** 
- Built an emotion classifier that goes beyond basic sentiment
- Tracks 8 different emotions: excited, frustrated, grateful, confused, satisfied, disappointed, hopeful, desperate
- Maps emotional intensity over time to spot trends and patterns

**Content Performance Analysis**
- Analyzes which posts drive the best engagement based on comment patterns
- Scores posts for "viral potential" by looking at user tagging and sharing behaviors  
- Automatically categorizes commenters by customer lifecycle stage (discovery, consideration, loyalty)
- Identifies posts that generate questions vs. posts that just get reactions

**Interactive Dashboards**
Built 4 main visualization dashboards using Plotly:
- **Comprehensive Overview**: All the key metrics in one place
- **Sentiment Deep Dive**: Sentiment trends over time and by customer segment  
- **Urgency & Priority**: Spots comments that need immediate attention
- **Emotion Heatmap**: Shows emotional patterns across different time periods

## Key Findings

### The Numbers
- **35.3%** of comments were positive vs only **3.39%** negative - pretty solid brand perception
- **"Excited"** was the most common emotion detected across all comments
- **362** comments asked questions while only **54** were seeking advice (6.7:1 ratio)
- **March 6th** had the highest positive sentiment spike, **April 2nd** was the lowest indicating a decline in positive sentiment.
- **54.5%** of urgent comments were still positive (customers stay optimistic even when they need help)
- **97.9%** of users are in the discovery phase - huge untapped potential
- **33.9%** of comments tagged other users for increased visibility

### What This Actually Means
TThis data shows that a brand with strong positive sentiment can grow more rapidly by focusing now to change the declining trajectory and increase customer engagement for growth.

## Recommendations

### 1. Target the Discovery Audience Better
With 97.9% of users in discovery mode, there's a clear opportunity to create content that moves people from "just browsing" to "I want to buy this." The high excitement levels show people are already interested and tagging their friends, drive that engagement!

### 2. Boost Content Positivity  
Current 52.4% positive sentiment is decent but could be higher. Since "excited" is already the dominant emotion, utilize that. More celebration posts, user wins, behind-the-scenes content that gets people pumped up.

### 3. Scale the Viral Winners
I found 30 posts with clear viral potential. Instead of guessing what works, analyze what made these posts successful and replicate those elements. This is basically a cheat sheet for future content strategy.

## Technical Implementation

### Tools & Technologies
- **Python** for all data processing and analysis
- **SQLite** for data storage and querying
- **Hugging Face Transformers** for sentiment and emotion analysis
- **Plotly** for interactive visualizations
- **Pandas** for data manipulation and analysis

### Architecture
The system is built in modular components that can run independently:

```
├── models/
│   ├── sentiment_analyzer.py    # Transformer-based sentiment scoring
│   ├── emotion_analyzer.py      # Multi-emotion classification  
│   ├── content_analyzer.py      # Post performance analysis
│   └── company_engagement.py    # Main orchestration class
├── data/
│   └── engagements.db          # SQLite database
├── dashboards/                 # Generated HTML visualizations
└── csv/                       # Raw data exports
```

### Helpful AI Assistance
Much thanks to Claude's robust ability to help me organize my ideas and focus a robust framework into a really interesting project.  I found that this could be something that i'd really enjoy expanding further and further given the neat ideas presented into the analytics.

### Performance Features
- **Batch processing** to handle large datasets efficiently
- **Caching** to avoid re-analyzing the same data
- **Modular design** makes it easy to add new analysis types
- **Export capabilities** for sharing results and further analysis

## What I Learned

Building this project taught me that it is fairly complex to analyze unstructured comment data compared to something more structured like sales records or even scientific literature to extract insights.  Comments often didn't follow any reasonable pattern so it is a challenge for any simplistic system to create adequate sentiment analysis or emotional analysis and have strong results.  A much more carefully crafted solution and a larger set of data could truly enhance this.

## Future Improvements

If I were to keep building on this:
- **Real-time processing** A Kafka event stream queue could be the foundation for a robust solution that could handle realtime analysis and social media monitoring with the confidence of data integrity
- **Expand sentiment/emotional analysis** Right now we do a decent job for one type of company but a much smarter analysis system would need to be in place to be able to handle additional companies and markets.
- **sqlite is great for proof of concept but we could expand this into a mySQL or PostgreSQL solution**
- **API/MCP** Development with Model Context Protocol could be an interesting way to connect the data to an LLM for additional insights and analysis.  A Product Manager with minimal SQL expertise could be empowered by asking smart questions and getting very intelligent responses based off the dataset
- **Competitive analysis** Ability to compare sentiment analysis across multiple brands might give our clients an edge
- **Predictive modeling** Simple trends are great but we should have some level of predictive modeling so clients can effectively pivot!
- **A/B testing framework** A host of testing could be used to increase the robustness of the analysis and tools but were unfortunately excluded due to time constraints.

## Files Structure
```
social-media-analytics/
├── README.md
├── main.py                     # Main execution script
├── models/
│   ├── __init__.py
│   ├── sentiment_analyzer.py
│   ├── emotion_analyzer.py
│   ├── content_analyzer.py
│   ├── company_engagement.py
│   └── visualization_analyzer.py
├── data/
│   └── engagements.db
├── csv/
│   └── engagements.csv
├── dashboards/
│   ├── sentiment_dashboard.html
│   ├── emotion_heatmap.html
│   ├── urgency_dashboard.html
│   └── comprehensive_dashboard.html
└── requirements.txt
```

## Contact

Craig Abernethy - craigabernethy85@gmail.com
