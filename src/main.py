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
from models.visualization_analyzer import VisualizationAnalyzer

def demonstrate_content_analysis(company):
    """Demonstrate the content analysis features"""
    # Print content analysis summary
    company.print_content_analysis_summary()
    
    # Get content insights
    insights = company.get_content_insights_summary()
    if 'error' not in insights:
        print(f"\n💡 Key Insights:")
        print(f"   📈 Total posts analyzed: {insights['total_posts_analyzed']}")
        print(f"   ⭐ Average engagement quality: {insights['avg_engagement_quality']:.3f}")
        print(f"   😊 Average positive sentiment: {insights['avg_positive_sentiment']:.1f}%")
        print(f"   🚀 Posts with viral potential: {insights['posts_with_high_viral_potential']}")
    
    # Generate strategy report
    print(f"\n📋 Generating strategy report...")
    strategy_report = company.generate_content_strategy_report()
    
    if 'error' not in strategy_report:
        print(f"   ✅ Strategy report generated successfully")
        print(f"   📊 Performance trend: {strategy_report['executive_summary']['content_performance_trend']}")
        print(f"   🎯 Key opportunity: {strategy_report['executive_summary']['key_opportunity']}")
        
        # Show top recommendations
        print(f"\n🎯 Top Recommendations:")
        for i, rec in enumerate(strategy_report['actionable_recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    else:
        print(f"   ❌ {strategy_report['error']}")

def generate_and_display_visualizations(company):
    """Generate and save all visualizations"""
    print("\n" + "="*60)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        # Create output directory for dashboards
        import os
        os.makedirs("dashboards", exist_ok=True)
        
        figures = {}
        
        # 1. Generate sentiment dashboard
        print("📊 Creating sentiment overview dashboard...")
        try:
            sentiment_fig = company.create_sentiment_overview_dashboard()
            figures['sentiment_dashboard'] = sentiment_fig
            
            # Save as HTML
            sentiment_path = company.save_dashboard_html(
                sentiment_fig, 
                "dashboards/sentiment_dashboard.html"
            )
            print(f"   ✅ Saved: {sentiment_path}")
        except Exception as e:
            print(f"   ❌ Error creating sentiment dashboard: {e}")
        
        # 2. Generate emotion heatmap
        print("🎭 Creating emotion heatmap...")
        try:
            emotion_fig = company.create_emotion_heatmap()
            figures['emotion_heatmap'] = emotion_fig
            
            # Save as HTML
            emotion_path = company.save_dashboard_html(
                emotion_fig, 
                "dashboards/emotion_heatmap.html"
            )
            print(f"   ✅ Saved: {emotion_path}")
        except Exception as e:
            print(f"   ❌ Error creating emotion heatmap: {e}")
        
        # 3. Generate urgency dashboard
        print("🚨 Creating urgency dashboard...")
        try:
            urgency_fig = company.create_urgency_dashboard()
            figures['urgency_dashboard'] = urgency_fig
            
            # Save as HTML
            urgency_path = company.save_dashboard_html(
                urgency_fig, 
                "dashboards/urgency_dashboard.html"
            )
            print(f"   ✅ Saved: {urgency_path}")
        except Exception as e:
            print(f"   ❌ Error creating urgency dashboard: {e}")
        
        # 4. Generate comprehensive dashboard
        print("📈 Creating comprehensive dashboard...")
        try:
            comprehensive_fig = company.create_comprehensive_dashboard()
            figures['comprehensive_dashboard'] = comprehensive_fig
            
            # Save as HTML
            comprehensive_path = company.save_dashboard_html(
                comprehensive_fig, 
                "dashboards/comprehensive_dashboard.html"
            )
            print(f"   ✅ Saved: {comprehensive_path}")
        except Exception as e:
            print(f"   ❌ Error creating comprehensive dashboard: {e}")
        
        # Summary
        if figures:
            print(f"\n✅ Successfully generated {len(figures)} visualizations!")
            print(f"📁 All dashboards saved in 'dashboards/' directory")
            print(f"🌐 Open the HTML files in your browser to view interactive dashboards")
            
            # List all generated files
            print(f"\n📋 Generated Files:")
            dashboard_files = [
                "dashboards/sentiment_dashboard.html",
                "dashboards/emotion_heatmap.html", 
                "dashboards/urgency_dashboard.html",
                "dashboards/comprehensive_dashboard.html"
            ]
            
            for file_path in dashboard_files:
                if os.path.exists(file_path):
                    print(f"   📊 {file_path}")
        else:
            print(f"❌ No visualizations were generated successfully")
        
        return figures
        
    except Exception as e:
        print(f"❌ Error in visualization generation: {e}")
        import traceback
        traceback.print_exc()
        return {}

def check_existing_analysis_data(company):
    """Check if we have existing analyzed data in the database"""
    print("🔍 Checking for existing analyzed data...")
    
    try:
        conn = company._ensure_connection()
        cursor = conn.cursor()
        
        # Check if comment_analysis table exists and has data
        cursor.execute('''
            SELECT COUNT(*) FROM comment_analysis ca
            JOIN engagement e ON ca.engagement_id = e.id
            JOIN company c ON e.company_id = c.id
            WHERE c.name = ?
        ''', (company.name,))
        analysis_count = cursor.fetchone()[0]
        
        # Check engagement data for context
        cursor.execute('''
            SELECT COUNT(*) FROM engagement e
            JOIN company c ON e.company_id = c.id  
            WHERE c.name = ?
        ''', (company.name,))
        engagement_count = cursor.fetchone()[0]
        
        print(f"   📊 Total engagement records: {engagement_count}")
        print(f"   🧠 Analyzed comment records: {analysis_count}")
        
        if engagement_count == 0:
            print("   ❌ No engagement data found")
            return False, "no_engagement_data"
        elif analysis_count == 0:
            print("   ⚠️  No analyzed data found - need to run analysis")
            return False, "no_analysis_data"
        elif analysis_count < engagement_count * 0.8:  # Less than 80% analyzed
            print(f"   ⚠️  Only {analysis_count}/{engagement_count} records analyzed ({analysis_count/engagement_count*100:.1f}%)")
            print("   💡 Consider re-running analysis for complete data")
            return True, "partial_analysis"
        else:
            print(f"   ✅ Good coverage: {analysis_count}/{engagement_count} records analyzed ({analysis_count/engagement_count*100:.1f}%)")
            return True, "complete_analysis"
            
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")
        return False, "error"

def check_data_availability(company):
    """Legacy function - kept for compatibility"""
    has_data, status = check_existing_analysis_data(company)
    return has_data

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    CSV_FILE = "csv/engagements.csv"
    DATABASE_FILE = "data/engagements.db"
    
    # Smart configuration - will be overridden based on existing data
    SKIP_COMMENT_ANALYSIS = False  # This will be set automatically based on existing data
    FORCE_RELOAD_CSV = False       # Set to True if you want to reload CSV even if DB exists
    
    print("🚀 Starting Enhanced Company Engagement Analysis Pipeline")
    print("="*80)
    
    # Run the script
    with CompanyEngagement("@treehut", "skincare", db_path=DATABASE_FILE) as company:
        
        # Step 1: Check if we should reload CSV data
        if FORCE_RELOAD_CSV or not os.path.exists(DATABASE_FILE):
            print("📥 Loading CSV data...")
            success = company.load_csv_to_database(CSV_FILE, overwrite=True)
            
            if not success:
                print("❌ Failed to load CSV data - exiting")
                exit(1)
        else:
            print("⏭️  Using existing database file")
        
        # Step 2: Smart check for existing analyzed data
        has_analysis_data, data_status = check_existing_analysis_data(company)
        
        # Step 3: Automatically decide whether to run analysis
        if has_analysis_data:
            print(f"✅ Found existing analyzed data ({data_status})")
            print("⚡ Skipping comment analysis to save time (10+ minutes)")
            SKIP_COMMENT_ANALYSIS = True
        else:
            print(f"⚠️  No analyzed data found ({data_status})")
            print("🧠 Will run comment analysis (this may take 10+ minutes)")
            SKIP_COMMENT_ANALYSIS = False
        
        # Step 4: Run comment analysis only if needed
        if not SKIP_COMMENT_ANALYSIS:
            print("\n🧠 Running comment analysis...")
            print("⏱️  This may take 10+ minutes depending on data size...")
            
            start_time = time.time()
            analysis_success = company.populate_comment_analysis(batch_size=100)
            end_time = time.time()
            
            if not analysis_success:
                print("❌ Comment analysis failed - cannot generate visualizations")
                exit(1)
            else:
                print(f"✅ Analysis completed in {(end_time - start_time)/60:.1f} minutes")
        else:
            print("⏭️  Skipping comment analysis - using existing data")
        
        # Step 5: Demonstrate content analysis
        print("\n📊 Running content analysis demonstration...")
        try:
            demonstrate_content_analysis(company)
        except Exception as e:
            print(f"❌ Content analysis demonstration failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 6: Export analysis results
        print("\n📄 Exporting analysis results...")
        try:
            export_path = company.export_content_analysis()
            if export_path:
                print(f"✅ Analysis exported to: {export_path}")
            else:
                print("⚠️  Export failed, but continuing...")
        except Exception as e:
            print(f"❌ Export failed: {e}")
        
        # Step 7: Generate and display visualizations
        print("\n🎨 Generating interactive visualizations...")
        try:
            figures = generate_and_display_visualizations(company)
            
            if figures:
                print(f"\n🎉 Pipeline completed successfully!")
                print(f"📊 Generated {len(figures)} interactive dashboards")
                print(f"📁 Check the 'dashboards/' directory for HTML files")
                print(f"🌐 Open HTML files in your browser to explore the data")
                
                # Optional: Print summary of what was generated
                print(f"\n📋 Available Dashboards:")
                dashboard_names = {
                    'sentiment_dashboard': 'Sentiment Analysis Overview',
                    'emotion_heatmap': 'Emotion Intensity Heatmap', 
                    'urgency_dashboard': 'Urgency & Priority Analysis',
                    'comprehensive_dashboard': 'Comprehensive Analytics Overview'
                }
                
                for key, name in dashboard_names.items():
                    if key in figures:
                        print(f"   ✅ {name}")
                    else:
                        print(f"   ❌ {name} (failed to generate)")
                        
                # Show file paths for easy access
                print(f"\n📂 Dashboard Files Created:")
                for file_name in ["sentiment_dashboard.html", "emotion_heatmap.html", 
                                "urgency_dashboard.html", "comprehensive_dashboard.html"]:
                    file_path = f"dashboards/{file_name}"
                    if os.path.exists(file_path):
                        print(f"   🌐 {os.path.abspath(file_path)}")
                        
            else:
                print("❌ No visualizations were generated")
                print("💡 Check the error messages above for troubleshooting")
                
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 8: Execution summary
        print(f"\n🏁 Pipeline execution completed!")
        print(f"\n📋 Execution Summary:")
        print(f"   💾 Database: {DATABASE_FILE}")
        print(f"   📊 CSV Reload: {'Yes' if FORCE_RELOAD_CSV or not os.path.exists(DATABASE_FILE) else 'Skipped'}")
        print(f"   🧠 Comment Analysis: {'Skipped (existing data)' if SKIP_COMMENT_ANALYSIS else 'Executed'}")
        print(f"   📈 Data Status: {data_status}")
        print(f"   🎨 Visualizations: {'Generated' if 'figures' in locals() and figures else 'Failed'}")
        print(f"   ⏱️  Total Time Saved: {'~10+ minutes' if SKIP_COMMENT_ANALYSIS and has_analysis_data else 'N/A'}")

# Quick function to just check your database status
def quick_status_check():
    """Quick function to check what's in your database without running anything"""
    print("🔍 Quick Database Status Check")
    print("="*50)
    
    if not os.path.exists("data/engagements.db"):
        print("❌ Database file not found: data/engagements.db")
        print("💡 Run the full pipeline to create it")
        return
    
    with CompanyEngagement("@treehut", "skincare", db_path="data/engagements.db") as company:
        has_data, status = check_existing_analysis_data(company)
        
        if has_data:
            print("✅ Ready for instant visualization generation!")
            print("💡 Run the main script - analysis will be skipped automatically")
        else:
            print(f"❌ Not ready: {status}")
            if status == "no_engagement_data":
                print("💡 Set FORCE_RELOAD_CSV = True to reload data")
            elif status == "no_analysis_data":
                print("💡 Run the main script to perform analysis")