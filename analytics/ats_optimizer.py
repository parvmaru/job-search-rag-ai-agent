"""
ATS Optimizer - Analyze resume for ATS (Applicant Tracking System) optimization

This module helps optimize resumes for ATS systems by:
1. Identifying missing keywords from job descriptions
2. Suggesting where to add keywords (skills/experience/projects sections)
3. Analyzing keyword density
4. Ranking keywords by importance
"""

from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from analytics.skill_gap import extract_keywords, extract_tech_keywords


def analyze_ats_match(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Analyze resume for ATS optimization against a job description.
    
    Args:
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        Dict with:
        - keywords_to_add: List[Dict] with 'keyword', 'frequency', 'importance_score', 'suggested_section'
        - suggested_sections: Dict mapping section names to lists of keywords
        - keyword_density: Dict with density stats
        - present_keywords: List of keywords found in resume
        - missing_keywords: List of keywords missing from resume
    """
    # Extract keywords from JD
    jd_keywords = extract_keywords(jd_text)
    jd_tech_keywords = extract_tech_keywords(jd_text)
    all_jd_keywords = jd_keywords.union(jd_tech_keywords)
    
    # Extract keywords from resume
    resume_keywords = extract_keywords(resume_text)
    resume_tech_keywords = extract_tech_keywords(resume_text)
    all_resume_keywords = resume_keywords.union(resume_tech_keywords)
    
    # Find missing keywords
    missing_keywords = all_jd_keywords - all_resume_keywords
    present_keywords = all_jd_keywords.intersection(all_resume_keywords)
    
    # Calculate keyword frequency in JD
    jd_text_lower = jd_text.lower()
    keyword_frequency = {}
    for keyword in all_jd_keywords:
        # Count occurrences (case-insensitive)
        count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', jd_text_lower))
        keyword_frequency[keyword] = count
    
    # Rank missing keywords by importance
    keywords_to_add = []
    for keyword in missing_keywords:
        freq = keyword_frequency.get(keyword, 0)
        
        # Calculate importance score (frequency + tech keyword bonus)
        importance_score = freq
        if keyword in jd_tech_keywords:
            importance_score += 5  # Tech keywords are more important
        
        # Determine suggested section
        suggested_section = _suggest_section(keyword, jd_text, resume_text)
        
        keywords_to_add.append({
            'keyword': keyword,
            'frequency': freq,
            'importance_score': importance_score,
            'suggested_section': suggested_section
        })
    
    # Sort by importance score (descending)
    keywords_to_add.sort(key=lambda x: x['importance_score'], reverse=True)
    
    # Group by suggested section
    suggested_sections = {
        'skills': [],
        'experience': [],
        'projects': [],
        'summary': []
    }
    
    for kw_data in keywords_to_add:
        section = kw_data['suggested_section']
        if section in suggested_sections:
            suggested_sections[section].append(kw_data['keyword'])
    
    # Calculate keyword density
    total_jd_keywords = len(all_jd_keywords)
    total_resume_keywords = len(all_resume_keywords)
    matched_keywords = len(present_keywords)
    
    keyword_density = {
        'jd_total': total_jd_keywords,
        'resume_total': total_resume_keywords,
        'matched_count': matched_keywords,
        'match_percentage': (matched_keywords / total_jd_keywords * 100) if total_jd_keywords > 0 else 0,
        'missing_count': len(missing_keywords),
        'missing_percentage': (len(missing_keywords) / total_jd_keywords * 100) if total_jd_keywords > 0 else 0
    }
    
    # Create data for visualization
    top_keywords_data = []
    for kw_data in keywords_to_add[:30]:  # Top 30 missing keywords
        top_keywords_data.append({
            'keyword': kw_data['keyword'],
            'frequency': kw_data['frequency'],
            'importance_score': kw_data['importance_score'],
            'present': False
        })
    
    # Add present keywords for comparison
    present_keywords_sorted = sorted(
        [(kw, keyword_frequency.get(kw, 0)) for kw in present_keywords],
        key=lambda x: x[1],
        reverse=True
    )[:20]  # Top 20 present keywords
    
    for kw, freq in present_keywords_sorted:
        top_keywords_data.append({
            'keyword': kw,
            'frequency': freq,
            'importance_score': freq + (5 if kw in jd_tech_keywords else 0),
            'present': True
        })
    
    return {
        'keywords_to_add': keywords_to_add,
        'suggested_sections': suggested_sections,
        'keyword_density': keyword_density,
        'present_keywords': list(present_keywords),
        'missing_keywords': list(missing_keywords),
        'top_keywords_data': top_keywords_data
    }


def _suggest_section(keyword: str, jd_text: str, resume_text: str) -> str:
    """
    Suggest which resume section a keyword should be added to.
    
    Args:
        keyword: The keyword to place
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        'skills', 'experience', 'projects', or 'summary'
    """
    keyword_lower = keyword.lower()
    
    # Check context in JD to determine section
    jd_lower = jd_text.lower()
    
    # Skills section indicators
    skills_indicators = [
        'skill', 'proficiency', 'expertise', 'knowledge', 'familiar', 
        'experience with', 'proficient in', 'knowledge of', 'understanding of'
    ]
    
    # Experience section indicators
    experience_indicators = [
        'experience', 'worked', 'developed', 'implemented', 'created',
        'built', 'designed', 'managed', 'led', 'achieved'
    ]
    
    # Projects section indicators
    projects_indicators = [
        'project', 'built', 'developed', 'created', 'designed',
        'portfolio', 'github', 'demo', 'prototype'
    ]
    
    # Find context around keyword in JD
    keyword_pos = jd_lower.find(keyword_lower)
    if keyword_pos != -1:
        # Get context (50 chars before and after)
        start = max(0, keyword_pos - 50)
        end = min(len(jd_lower), keyword_pos + len(keyword) + 50)
        context = jd_lower[start:end]
        
        # Check for section indicators
        for indicator in skills_indicators:
            if indicator in context:
                return 'skills'
        
        for indicator in projects_indicators:
            if indicator in context:
                return 'projects'
        
        for indicator in experience_indicators:
            if indicator in context:
                return 'experience'
    
    # Default logic based on keyword characteristics
    # Tech keywords usually go in skills
    tech_patterns = [
        r'\b(python|java|javascript|react|node|sql|aws|docker|kubernetes|git|linux|windows|macos)\b',
        r'\b(framework|library|tool|technology|platform|language)\b'
    ]
    
    for pattern in tech_patterns:
        if re.search(pattern, keyword_lower):
            return 'skills'
    
    # Action verbs usually go in experience
    action_verbs = [
        'develop', 'create', 'build', 'design', 'implement', 'manage',
        'lead', 'optimize', 'improve', 'analyze', 'solve'
    ]
    
    if any(verb in keyword_lower for verb in action_verbs):
        return 'experience'
    
    # Default to skills for most keywords
    return 'skills'


def create_keywords_chart(keywords_data: List[Dict[str, Any]], top_n: int = 20) -> Any:
    """
    Create a Plotly bar chart showing top keywords and whether they're present in resume.
    
    Args:
        keywords_data: List of keyword dicts with 'keyword', 'frequency', 'importance_score', 'present'
        top_n: Number of top keywords to show
        
    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go
    import pandas as pd
    
    # Sort by importance score and take top N
    sorted_data = sorted(keywords_data, key=lambda x: x['importance_score'], reverse=True)[:top_n]
    
    # Separate present and missing
    keywords = [d['keyword'] for d in sorted_data]
    frequencies = [d['frequency'] for d in sorted_data]
    present = [d['present'] for d in sorted_data]
    
    # Create colors based on presence
    colors = ['#4ecdc4' if p else '#ff6b6b' for p in present]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=frequencies,
            y=keywords,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=[f"✓ Present" if p else "✗ Missing" for p in present],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Frequency: %{x}<br>Status: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Keywords: Present vs Missing',
        xaxis_title='Frequency in Job Description',
        yaxis_title='Keywords',
        yaxis={
            'categoryorder': 'total ascending',
            'gridcolor': 'rgba(255,255,255,0.1)'
        },
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa', size=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

