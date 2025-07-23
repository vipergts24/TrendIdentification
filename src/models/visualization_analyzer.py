
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not installed. Install with: pip install plotly")


class VisualizationAnalyzer:
    """
    Comprehensive visualization system for social media engagement data.
    
    Creates interactive charts and dashboards for sentiment analysis, emotion detection,
    content performance, and customer insights using Plotly.
    """
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize the visualization analyzer.
        
        Args:
            db_connection: Active SQLite database connection
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")
        
        self.connection = db_connection
        self.cursor = db_connection.cursor()
        
        # Set default Plotly theme
        pio.templates.default = "plotly_white"
        
        # Color schemes for consistent branding
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
    
    def create_sentiment_overview_dashboard(self, company_name: str = "@treehut") -> go.Figure:
        """
        Create comprehensive sentiment analysis dashboard.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            go.Figure: Interactive dashboard with sentiment metrics
        """
        # Get sentiment data
        sentiment_data = self._get_sentiment_data(company_name)
        
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
        daily_sentiment = self._get_daily_sentiment_data(company_name)
        
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
        lifecycle_sentiment = self._get_lifecycle_sentiment_data(company_name)
        
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
            title=f'Sentiment Analysis Dashboard - {company_name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_emotion_heatmap(self, company_name: str = "@treehut") -> go.Figure:
        """
        Create emotion intensity heatmap across different content and time periods.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            go.Figure: Interactive emotion heatmap
        """
        emotion_data = self._get_emotion_heatmap_data(company_name)
        
        if emotion_data.empty:
            return self._create_no_data_figure("No emotion data available")
        
        # Prepare heatmap data
        emotions = ['excited', 'frustrated', 'grateful', 'confused', 'satisfied', 'disappointed', 'hopeful', 'desperate']
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=emotion_data[emotions].values,
            x=emotions,
            y=emotion_data.index,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{x}</b><br>Period: %{y}<br>Intensity: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Emotion Intensity", titleside="right")
        ))
        
        fig.update_layout(
            title=f'Emotion Intensity Heatmap - {company_name}',
            xaxis_title='Emotions',
            yaxis_title='Time Period',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_content_performance_dashboard(self) -> go.Figure:
        """
        Create comprehensive content performance dashboard.
        
        Returns:
            go.Figure: Interactive content performance dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations")
        
        content_data = self._get_content_performance_data()
        
        if content_data.empty:
            return self._create_no_data_figure("No content performance data available")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Engagement Quality vs. Total Comments',
                'Sentiment Distribution by Post Performance',
                'Top Emotions by High-Performing Posts',
                'Viral Score vs. Engagement Quality'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Engagement Quality vs Total Comments (Bubble Chart)
        if all(col in content_data.columns for col in ['total_comments', 'engagement_quality_score', 'positive_sentiment_pct', 'viral_score', 'media_id']):
            fig.add_trace(
                go.Scatter(
                    x=content_data['total_comments'],
                    y=content_data['engagement_quality_score'],
                    mode='markers',
                    marker=dict(
                        size=content_data['positive_sentiment_pct'].fillna(10),  # Handle NaN values
                        color=content_data['viral_score'].fillna(0),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Viral Score", x=0.45),
                        sizemode='diameter',
                        sizeref=2.*max(content_data['positive_sentiment_pct'].fillna(1))/(40.**2),
                        sizemin=4,
                        line=dict(width=1, color='white')
                    ),
                    text=content_data['media_id'],
                    hovertemplate='<b>%{text}</b><br>Comments: %{x}<br>Quality: %{y:.3f}<br>Positive Sentiment: %{marker.size:.1f}%<br>Viral Score: %{marker.color:.3f}<extra></extra>',
                    name='Posts'
                ),
                row=1, col=1
            )
        
        # 2. Sentiment by Performance Level
        if 'engagement_quality_score' in content_data.columns:
            performance_levels = pd.cut(content_data['engagement_quality_score'], 
                                    bins=[0, 0.3, 0.6, 1.0], 
                                    labels=['Low', 'Medium', 'High'])
            
            sentiment_cols = [col for col in ['positive_sentiment_pct', 'negative_sentiment_pct', 'neutral_sentiment_pct'] 
                            if col in content_data.columns]
            
            if sentiment_cols:
                sentiment_by_performance = content_data.groupby(performance_levels)[sentiment_cols].mean()
                
                for sentiment in sentiment_cols:
                    sentiment_name = sentiment.replace('_sentiment_pct', '').capitalize()
                    color = self.colors['sentiment'].get(sentiment.replace('_sentiment_pct', ''), self.colors['primary'])
                    
                    fig.add_trace(
                        go.Bar(
                            x=sentiment_by_performance.index,
                            y=sentiment_by_performance[sentiment],
                            name=sentiment_name,
                            marker_color=color,
                            hovertemplate=f'<b>{sentiment_name}</b><br>Performance: %{{x}}<br>Avg %: %{{y:.1f}}<extra></extra>'
                        ),
                        row=1, col=2
                    )
        
        # 3. Top Emotions in High-Performing Posts
        if 'engagement_quality_score' in content_data.columns and 'top_emotion' in content_data.columns:
            high_performers = content_data[content_data['engagement_quality_score'] > 0.6]
            if not high_performers.empty:
                top_emotions = high_performers['top_emotion'].value_counts().head(6)
                
                fig.add_trace(
                    go.Bar(
                        x=top_emotions.index,
                        y=top_emotions.values,
                        name="High Performers",
                        marker_color=[self.colors['emotions'].get(emotion, self.colors['primary']) for emotion in top_emotions.index],
                        showlegend=False,
                        hovertemplate='<b>%{x}</b><br>High-Performing Posts: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Viral Score vs Engagement Quality
        if all(col in content_data.columns for col in ['engagement_quality_score', 'viral_score', 'positive_sentiment_pct', 'media_id']):
            fig.add_trace(
                go.Scatter(
                    x=content_data['engagement_quality_score'],
                    y=content_data['viral_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=content_data['positive_sentiment_pct'].fillna(0),
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Positive %", x=1.02),
                        line=dict(width=1, color='white')
                    ),
                    text=content_data['media_id'],
                    hovertemplate='<b>%{text}</b><br>Engagement Quality: %{x:.3f}<br>Viral Score: %{y:.3f}<br>Positive Sentiment: %{marker.color:.1f}%<extra></extra>',
                    name='Posts'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Content Performance Dashboard - {self.name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Total Comments", row=1, col=1)
        fig.update_yaxes(title_text="Engagement Quality", row=1, col=1)
        fig.update_xaxes(title_text="Performance Level", row=1, col=2)
        fig.update_yaxes(title_text="Sentiment %", row=1, col=2)
        fig.update_xaxes(title_text="Emotion", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Engagement Quality", row=2, col=2)
        fig.update_yaxes(title_text="Viral Score", row=2, col=2)
        
        return fig
        
    def create_customer_journey_visualization(self, company_name: str = "@treehut") -> go.Figure:
        """
        Create customer journey and lifecycle analysis visualization.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            go.Figure: Interactive customer journey visualization
        """
        journey_data = self._get_customer_journey_data(company_name)
        
        if journey_data.empty:
            return self._create_no_data_figure("No customer journey data available")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Customer Lifecycle Distribution',
                'Urgency Levels by Lifecycle Stage',
                'Emotions by Customer Lifecycle',
                'Sentiment by Lifecycle Stage'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Lifecycle Distribution
        lifecycle_counts = journey_data['customer_lifecycle_stage'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=lifecycle_counts.index,
                values=lifecycle_counts.values,
                name="Lifecycle Distribution",
                marker_colors=[self.colors['lifecycle'][stage] for stage in lifecycle_counts.index],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Urgency by Lifecycle
        urgency_lifecycle = pd.crosstab(journey_data['customer_lifecycle_stage'], journey_data['urgency_level'])
        
        for urgency in ['low', 'medium', 'high']:
            if urgency in urgency_lifecycle.columns:
                fig.add_trace(
                    go.Bar(
                        x=urgency_lifecycle.index,
                        y=urgency_lifecycle[urgency],
                        name=f'{urgency.capitalize()} Urgency',
                        marker_color=self.colors['urgency'][urgency],
                        hovertemplate=f'<b>{urgency.capitalize()} Urgency</b><br>Stage: %{{x}}<br>Count: %{{y}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # 3. Emotions by Lifecycle (top emotions only)
        emotion_lifecycle_data = self._get_emotion_by_lifecycle_data(company_name)
        
        if not emotion_lifecycle_data.empty:
            # Get top 4 emotions
            top_emotions = emotion_lifecycle_data.sum().nlargest(4).index
            
            for emotion in top_emotions:
                fig.add_trace(
                    go.Bar(
                        x=emotion_lifecycle_data.index,
                        y=emotion_lifecycle_data[emotion],
                        name=emotion.capitalize(),
                        marker_color=self.colors['emotions'].get(emotion, self.colors['primary']),
                        hovertemplate=f'<b>{emotion.capitalize()}</b><br>Stage: %{{x}}<br>Avg Score: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 4. Sentiment by Lifecycle
        sentiment_lifecycle = journey_data.groupby('customer_lifecycle_stage')[['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']].mean()
        
        for sentiment in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']:
            sentiment_name = sentiment.replace('sentiment_', '').capitalize()
            color = self.colors['sentiment'][sentiment.replace('sentiment_', '')]
            
            fig.add_trace(
                go.Bar(
                    x=sentiment_lifecycle.index,
                    y=sentiment_lifecycle[sentiment],
                    name=sentiment_name,
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=f'<b>{sentiment_name}</b><br>Stage: %{{x}}<br>Avg Score: %{{y:.3f}}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Customer Journey Analysis - {company_name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_urgency_priorities_dashboard(self, company_name: str = "@treehut") -> go.Figure:
        """
        Create urgency and priority management dashboard.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            go.Figure: Interactive urgency priorities dashboard
        """
        urgency_data = self._get_urgency_data(company_name)
        
        if urgency_data.empty:
            return self._create_no_data_figure("No urgency data available")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Urgency Distribution Over Time',
                'High-Urgency Comments by Sentiment',
                'Urgency vs. Engagement Quality',
                'Response Priority Matrix'
            ),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Urgency Over Time
        daily_urgency = self._get_daily_urgency_data(company_name)
        
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
        
        # 3. Urgency vs Engagement Quality
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
        
        # 4. Priority Matrix (Urgency vs Sentiment)
        # Create priority scores
        priority_scores = urgency_data.copy()
        urgency_map = {'low': 1, 'medium': 2, 'high': 3}
        priority_scores['urgency_numeric'] = priority_scores['urgency_level'].map(urgency_map)
        priority_scores['priority_score'] = priority_scores['urgency_numeric'] * (1 - priority_scores['sentiment_positive'])
        
        fig.add_trace(
            go.Scatter(
                x=priority_scores['urgency_numeric'],
                y=priority_scores['sentiment_negative'],
                mode='markers',
                marker=dict(
                    size=priority_scores['priority_score'] * 5,
                    color=priority_scores['priority_score'],
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
        
        # Update layout
        fig.update_layout(
            title=f'Urgency & Priority Dashboard - {company_name}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Word Count", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Urgency Level (1=Low, 3=High)", row=2, col=2)
        fig.update_yaxes(title_text="Negative Sentiment Score", row=2, col=2)
        
        return fig
    
    def create_viral_potential_analysis(self, company_name: str = "@treehut") -> go.Figure:
        """
        Create viral potential and sharing analysis dashboard.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            go.Figure: Interactive viral potential analysis
        """
        viral_data = self._get_viral_potential_data(company_name)
        
        if viral_data.empty:
            return self._create_no_data_figure("No viral potential data available")
        
        # Create the visualization
        fig = go.Figure()
        
        # Bubble chart: Engagement Quality vs Positive Sentiment, sized by viral score
        fig.add_trace(
            go.Scatter(
                x=viral_data['engagement_quality_score'],
                y=viral_data['positive_sentiment_pct'],
                mode='markers',
                marker=dict(
                    size=viral_data['viral_score'] * 100,  # Scale up for visibility
                    color=viral_data['viral_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Viral Score"),
                    sizemode='diameter',
                    sizeref=2.*max(viral_data['viral_score'] * 100)/(40.**2),
                    sizemin=4,
                    line=dict(width=2, color='white')
                ),
                text=viral_data['media_id'],
                hovertemplate='<b>%{text}</b><br>Engagement Quality: %{x:.3f}<br>Positive Sentiment: %{y:.1f}%<br>Viral Score: %{marker.color:.3f}<br>User Tagging Rate: ' + viral_data['user_tagging_rate'].astype(str) + '%<extra></extra>',
                name='Posts'
            )
        )
        
        # Add quadrant lines
        median_quality = viral_data['engagement_quality_score'].median()
        median_sentiment = viral_data['positive_sentiment_pct'].median()
        
        fig.add_hline(y=median_sentiment, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_quality, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add annotations for quadrants
        fig.add_annotation(x=0.1, y=90, text="High Sentiment<br>Low Quality", showarrow=False, bgcolor="rgba(255,255,255,0.8)")
        fig.add_annotation(x=0.9, y=90, text="High Sentiment<br>High Quality<br>(VIRAL ZONE)", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
        fig.add_annotation(x=0.1, y=10, text="Low Sentiment<br>Low Quality", showarrow=False, bgcolor="rgba(255,255,255,0.8)")
        fig.add_annotation(x=0.9, y=10, text="High Quality<br>Low Sentiment", showarrow=False, bgcolor="rgba(255,255,255,0.8)")
        
        fig.update_layout(
            title=f'Viral Potential Analysis - {company_name}<br><sub>Bubble size represents viral score (user tagging rate)</sub>',
            xaxis_title='Engagement Quality Score',
            yaxis_title='Positive Sentiment Percentage',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def save_dashboard_html(self, fig: go.Figure, filename: str, auto_open: bool = False) -> str:
        """
        Save dashboard as interactive HTML file.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            auto_open: Whether to automatically open in browser
            
        Returns:
            str: Path to saved file
        """
        if not filename.endswith('.html'):
            filename += '.html'
        
        fig.write_html(filename, auto_open=auto_open)
        return filename
    
    # Helper methods for data retrieval
    def _get_sentiment_data(self, company_name: str) -> pd.DataFrame:
        """Get sentiment analysis data."""
        query = '''
            SELECT ca.sentiment_label, ca.sentiment_confidence, ca.sentiment_positive,
                   ca.sentiment_negative, ca.sentiment_neutral, ca.customer_lifecycle_stage,
                   e.timestamp
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        '''
        return pd.read_sql_query(query, self.connection, params=(company_name,))
    
    def _get_daily_sentiment_data(self, company_name: str) -> pd.DataFrame:
        """Get daily aggregated sentiment data."""
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
        df = pd.read_sql_query(query, self.connection, params=(company_name,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def _get_lifecycle_sentiment_data(self, company_name: str) -> pd.DataFrame:
        """Get sentiment data by customer lifecycle stage."""
        query = '''
            SELECT ca.customer_lifecycle_stage,
                   ca.sentiment_label,
                   COUNT(*) as count
            FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
            GROUP BY ca.customer_lifecycle_stage, ca.sentiment_label
        '''