#!/usr/bin/env python3
"""
📊 YouTube Analytics Integration for Quality Checker
Adds CTR/AVD tracking to Quality Checker reports
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

BITTRADER = Path("/home/enderj/.openclaw/workspace/bittrader")
DATA_DIR = BITTRADER / "agents/data"
CREDENTIALS_FILE = Path("/home/enderj/.openclaw/workspace/memory/youtube_credentials.json")


def get_video_analytics(video_id: str) -> dict:
    """Get CTR and AVD metrics for a video from YouTube Analytics"""
    
    if not CREDENTIALS_FILE.exists():
        return {"error": "YouTube credentials not found"}
    
    try:
        creds = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE))
        youtube = build('youtubeAnalytics', 'v2', credentials=creds)
        youtube_data = build('youtube', 'v3', credentials=creds)
        
        # Get video basic info
        video_response = youtube_data.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return {"error": "Video not found"}
        
        video = video_response['items'][0]
        
        # Get analytics data (last 7 days)
        analytics_response = youtube.reports().query(
            ids='channel==MINE',
            metrics='views,estimatedMinutesWatched,averageViewDuration,clickThroughRate,likes,comments',
            filters=f'video=={video_id}',
            startDate=(datetime.now() - __import__('datetime').timedelta(days=7)).strftime('%Y-%m-%d'),
            endDate=datetime.now().strftime('%Y-%m-%d')
        ).execute()
        
        # Extract metrics
        stats = video['statistics']
        
        result = {
            "video_id": video_id,
            "title": video['snippet']['title'],
            "published_at": video['snippet']['publishedAt'],
            "duration": video['contentDetails']['duration'],
            "metrics": {
                "views": int(stats.get('viewCount', 0)),
                "likes": int(stats.get('likeCount', 0)),
                "comments": int(stats.get('commentCount', 0)),
                "ctr": None,  # From analytics
                "avd_seconds": None,  # From analytics
                "avd_percentage": None
            }
        }
        
        # Parse analytics if available
        if 'rows' in analytics_response and analytics_response['rows']:
            row = analytics_response['rows'][0]
            col_headers = analytics_response['columnHeaders']
            
            for i, header in enumerate(col_headers):
                if header['name'] == 'clickThroughRate':
                    result['metrics']['ctr'] = row[i]
                elif header['name'] == 'averageViewDuration':
                    result['metrics']['avd_seconds'] = row[i]
        
        # Calculate AVD percentage (if we have duration and AVD)
        if result['metrics']['avd_seconds']:
            # Parse ISO 8601 duration (PT#M#S)
            duration_str = video['contentDetails']['duration']
            import re
            match = re.match(r'PT(?:(\d+)M)?(?:(\d+)S)?', duration_str)
            if match:
                minutes = int(match.group(1) or 0)
                seconds = int(match.group(2) or 0)
                total_duration = minutes * 60 + seconds
                
                if total_duration > 0:
                    result['metrics']['avd_percentage'] = round(
                        (result['metrics']['avd_seconds'] / total_duration) * 100, 2
                    )
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


def score_video_performance(metrics: dict) -> dict:
    """Score video based on MrBeast performance benchmarks"""
    
    if "error" in metrics:
        return metrics
    
    score = 0
    max_score = 100
    issues = []
    recommendations = []
    
    # CTR scoring (MrBeast benchmark: >10%)
    ctr = metrics.get('metrics', {}).get('ctr')
    if ctr:
        if ctr >= 0.10:
            score += 30
        elif ctr >= 0.07:
            score += 20
            issues.append("CTR below MrBeast benchmark (7% vs 10%+)")
            recommendations.append("Test more expressive thumbnail faces")
        elif ctr >= 0.05:
            score += 10
            issues.append("CTR low (5-7%)")
            recommendations.append("Redesign thumbnail with better contrast and faces")
        else:
            issues.append("CTR very low (<5%)")
            recommendations.append("Complete thumbnail redesign needed")
    
    # AVD scoring (MrBeast benchmark: >70%)
    avd_pct = metrics.get('metrics', {}).get('avd_percentage')
    if avd_pct:
        if avd_pct >= 70:
            score += 30
        elif avd_pct >= 50:
            score += 20
            issues.append("AVD below MrBeast benchmark (50-70% vs 70%+)")
            recommendations.append("Add more re-engagement hooks every 3 min")
        elif avd_pct >= 30:
            score += 10
            issues.append("AVD low (30-50%)")
            recommendations.append("Improve opening hook and add plot twists")
        else:
            issues.append("AVD very low (<30%)")
            recommendations.append("Restructure content with MrBeast retention tactics")
    
    # Engagement scoring
    views = metrics.get('metrics', {}).get('views', 0)
    likes = metrics.get('metrics', {}).get('likes', 0)
    comments = metrics.get('metrics', {}).get('comments', 0)
    
    if views > 0:
        like_rate = likes / views
        comment_rate = comments / views
        
        # Good engagement: >5% like rate, >0.5% comment rate
        if like_rate >= 0.05:
            score += 20
        elif like_rate >= 0.03:
            score += 15
        else:
            issues.append("Low like rate")
        
        if comment_rate >= 0.005:
            score += 20
        elif comment_rate >= 0.002:
            score += 15
        else:
            issues.append("Low comment rate")
            recommendations.append("Add CTA question at end of video")
    
    return {
        "video_id": metrics['video_id'],
        "title": metrics['title'],
        "performance_score": score,
        "max_score": max_score,
        "grade": "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "F",
        "issues": issues,
        "recommendations": recommendations,
        "raw_metrics": metrics
    }


if __name__ == "__main__":
    print("=" * 80)
    print("📊 YOUTUBE ANALYTICS - QUALITY CHECKER INTEGRATION")
    print("=" * 80)
    print()
    
    # Test with sample video
    test_video_id = "ciiiE0klMBg"  # AKT video
    
    print(f"📊 Fetching analytics for: {test_video_id}")
    print()
    
    metrics = get_video_analytics(test_video_id)
    
    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        print()
        print("⚠️ Note: YouTube Analytics requires OAuth with youtube.readonly scope")
        print("Current credentials may not have the required permissions.")
    else:
        print(f"📹 Video: {metrics['title']}")
        print(f"📅 Published: {metrics['published_at']}")
        print()
        print("📊 METRICS:")
        print(f"  Views: {metrics['metrics']['views']:,}")
        print(f"  Likes: {metrics['metrics']['likes']:,}")
        print(f"  Comments: {metrics['metrics']['comments']:,}")
        if metrics['metrics']['ctr']:
            print(f"  CTR: {metrics['metrics']['ctr']:.2%}")
        if metrics['metrics']['avd_percentage']:
            print(f"  AVD: {metrics['metrics']['avd_percentage']:.1f}%")
        print()
        
        # Score performance
        performance = score_video_performance(metrics)
        print(f"🎯 PERFORMANCE SCORE: {performance['performance_score']}/{performance['max_score']}")
        print(f"   Grade: {performance['grade']}")
        
        if performance['issues']:
            print()
            print("⚠️ ISSUES:")
            for issue in performance['issues']:
                print(f"  • {issue}")
        
        if performance['recommendations']:
            print()
            print("💡 RECOMMENDATIONS:")
            for rec in performance['recommendations']:
                print(f"  • {rec}")
        
        # Save report
        report_file = DATA_DIR / f"analytics_report_{test_video_id}.json"
        report_file.write_text(json.dumps(performance, indent=2))
        print()
        print(f"💾 Report saved: {report_file}")
