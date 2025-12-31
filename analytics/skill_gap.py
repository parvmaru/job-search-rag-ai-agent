"""
Skill Gap Analytics - Keyword extraction and visualization

Fast keyword-based analysis using TF-IDF and simple text matching.
This is much faster than LLM-based analysis.
"""

from typing import List, Dict, Set, Tuple
import re
from collections import Counter
import numpy as np


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract keywords from text using simple word-based extraction.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        
    Returns:
        Set of lowercase keywords
    """
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it',
        'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your', 'i', 'my',
        'me', 'he', 'she', 'his', 'her', 'from', 'as', 'if', 'when', 'where',
        'what', 'which', 'who', 'how', 'why', 'about', 'into', 'through', 'during'
    }
    
    # Filter by length and stop words
    keywords = {w for w in words if len(w) >= min_length and w not in stop_words}
    return keywords


def extract_tech_keywords(text: str) -> Set[str]:
    """
    Extract technical keywords (skills, tools, technologies).
    
    Common tech patterns:
    - Programming languages
    - Frameworks
    - Tools and platforms
    - Methodologies
    """
    tech_keywords = set()
    text_lower = text.lower()
    
    # Common tech keyword patterns (case-insensitive matching)
    tech_patterns = [
        # Programming languages
        r'\b(python|java|javascript|typescript|go|rust|c\+\+|c#|php|ruby|swift|kotlin|scala|r|matlab|perl|bash|shell|powershell)\b',
        # Frameworks & Libraries
        r'\b(react|angular|vue|django|flask|spring|express|node|next|nuxt|laravel|rails|asp\.net|tensorflow|pytorch|pandas|numpy|scikit|keras)\b',
        # Databases
        r'\b(mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb|sqlite|oracle|sql server|nosql|postgres)\b',
        # Cloud & DevOps
        r'\b(aws|azure|gcp|docker|kubernetes|terraform|jenkins|gitlab|github|ci/cd|devops|s3|ec2|lambda|cloudformation)\b',
        # Tools & Platforms
        r'\b(git|jira|confluence|tableau|powerbi|excel|spark|hadoop|kafka|rabbitmq|airflow|databricks|snowflake|redshift)\b',
        # Methodologies & Concepts
        r'\b(agile|scrum|kanban|lean|waterfall|tdd|bdd|microservices|rest|graphql|api|ml|ai|machine learning|deep learning)\b',
        # Additional tech terms
        r'\b(html|css|json|xml|yaml|toml|linux|unix|windows|macos|ios|android|ios|android)\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        tech_keywords.update(matches)
    
    # Extract capitalized tech terms (common in resumes/JDs)
    # Look for patterns like "Python", "React", "AWS", etc.
    capitalized = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', text)
    # Filter out common non-tech capitalized words (company names, common words, etc.)
    non_tech_words = {
        'The', 'This', 'That', 'These', 'Those', 'Your', 'You', 'We', 'Our', 'Their', 
        'Company', 'Team', 'Project', 'Work', 'Will', 'Please', 'Support', 'Help',
        'Eligible', 'Range', 'Pay', 'Remote', 'Develop', 'Us', 'If', 'To', 'Of',
        'Airbnb', 'Petco', 'Trajectory'  # Common company names
    }
    tech_caps = [w.lower() for w in capitalized if w not in non_tech_words and len(w) >= 2]
    tech_keywords.update(tech_caps)
    
    # Extract acronyms (often tech terms) - but filter out common non-tech acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    non_tech_acronyms = {'US', 'JD', 'PDF', 'API', 'URL', 'HTTP', 'HTTPS', 'JSON', 'XML', 'YAML'}  # Keep API, JSON, XML, YAML as they're tech
    tech_acronyms = [a.lower() for a in acronyms if a not in non_tech_acronyms or a in {'API', 'JSON', 'XML', 'YAML', 'REST', 'SQL', 'AWS', 'GCP', 'CI', 'CD', 'ML', 'AI', 'TDD', 'BDD'}]
    tech_keywords.update(tech_acronyms)
    
    # Remove common stop words and non-tech words that might have been captured
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 
        'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 
        'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'she', 'to', 'of', 
        'if', 'us', 'work', 'will', 'help', 'please', 'support', 'range', 'pay', 
        'remote', 'develop', 'eligible', 'airbnb', 'petco', 'trajectory'
    }
    tech_keywords = tech_keywords - stop_words
    
    return tech_keywords


def compute_skill_gap(jd_text: str, resume_text: str) -> Dict[str, any]:
    """
    Compute skill gap analysis between JD and Resume.
    
    IMPORTANT: Match score is based ONLY on tech keywords (100% weight).
    General keywords are excluded from match score calculation.
    
    Args:
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        Dict with overlap_percentage, missing_keywords, present_keywords, etc.
    """
    # Extract ONLY tech keywords - this is what matters for match score
    jd_keywords = extract_tech_keywords(jd_text)
    resume_keywords = extract_tech_keywords(resume_text)
    
    # Calculate tech keyword overlap
    tech_overlap = jd_keywords.intersection(resume_keywords)
    tech_missing = jd_keywords - resume_keywords
    
    # Match score is 100% based on tech keywords ONLY
    tech_total = len(jd_keywords) if len(jd_keywords) > 0 else 1
    tech_overlap_count = len(tech_overlap)
    tech_match_ratio = tech_overlap_count / tech_total if tech_total > 0 else 0
    
    # Match score = percentage of tech keywords that match
    # Use aggressive scoring: if even 1 tech keyword matches, give meaningful score
    if tech_total > 0:
        # Base score from match ratio
        overlap_percentage = tech_match_ratio * 100
        
        # Aggressive boost logic: Ensure scores reflect actual tech keyword matches
        if tech_match_ratio >= 0.7:  # 70%+ match
            overlap_percentage = max(overlap_percentage, 75)
        elif tech_match_ratio >= 0.5:  # 50-70% match
            overlap_percentage = max(overlap_percentage, 60)
        elif tech_match_ratio >= 0.4:  # 40-50% match
            overlap_percentage = max(overlap_percentage, 50)
        elif tech_match_ratio >= 0.3:  # 30-40% match
            overlap_percentage = max(overlap_percentage, 40)
        elif tech_match_ratio >= 0.2:  # 20-30% match
            overlap_percentage = max(overlap_percentage, 30)
        elif tech_match_ratio >= 0.1:  # 10-20% match
            overlap_percentage = max(overlap_percentage, 20)
        elif tech_match_ratio > 0:  # Any match
            overlap_percentage = max(overlap_percentage, 15)
        
        # Additional boost: if absolute number of matches is high, boost further
        if tech_overlap_count >= 10:
            overlap_percentage = max(overlap_percentage, 60)
        elif tech_overlap_count >= 5:
            overlap_percentage = max(overlap_percentage, 40)
        elif tech_overlap_count >= 3:
            overlap_percentage = max(overlap_percentage, 25)
    else:
        overlap_percentage = 0
    
    missing_percentage = 100 - overlap_percentage
    
    # Get top missing tech keywords by frequency in JD
    # STRICT filtering: Only include keywords that match tech patterns
    jd_text_lower = jd_text.lower()
    missing_freq = {}
    
    # Known tech keywords list for strict filtering
    known_tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell', 'powershell',
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node', 'next', 'nuxt', 'laravel', 'rails', 'asp.net', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit', 'keras',
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'sqlite', 'oracle', 'sql', 'nosql', 'postgres',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github', 'ci/cd', 'devops', 's3', 'ec2', 'lambda', 'cloudformation',
        'git', 'jira', 'confluence', 'tableau', 'powerbi', 'excel', 'spark', 'hadoop', 'kafka', 'rabbitmq', 'airflow', 'databricks', 'snowflake', 'redshift',
        'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'tdd', 'bdd', 'microservices', 'rest', 'graphql', 'api', 'ml', 'ai', 'machine learning', 'deep learning',
        'html', 'css', 'json', 'xml', 'yaml', 'toml', 'linux', 'unix', 'windows', 'macos', 'ios', 'android'
    }
    
    for keyword in tech_missing:
        kw_lower = keyword.lower()
        # Only include if it's in known tech keywords or matches tech patterns
        if kw_lower in known_tech_keywords or any(tech in kw_lower or kw_lower in tech for tech in known_tech_keywords):
            # Count occurrences (case-insensitive)
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', jd_text_lower))
            if count > 0:
                missing_freq[keyword] = count
        # Also allow tech acronyms (2-4 uppercase letters)
        elif re.match(r'^[A-Z]{2,4}$', keyword) and keyword not in ['US', 'JD', 'PDF', 'URL', 'HTTP', 'HTTPS']:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', jd_text))
            if count > 0:
                missing_freq[keyword] = count
    
    # Sort by frequency
    top_missing = sorted(missing_freq.items(), key=lambda x: x[1], reverse=True)
    
    # For reference: also get general keywords (but don't use for match score)
    jd_general = extract_keywords(jd_text)
    resume_general = extract_keywords(resume_text)
    jd_all = jd_keywords.union(jd_general)
    resume_all = resume_keywords.union(resume_general)
    overlap_keywords = jd_all.intersection(resume_all)
    
    return {
        'overlap_percentage': round(overlap_percentage, 1),
        'missing_percentage': round(missing_percentage, 1),
        'simple_overlap_percentage': round((len(overlap_keywords) / len(jd_all) * 100) if len(jd_all) > 0 else 0, 1),  # For reference only
        'overlap_keywords': list(tech_overlap),  # Only tech keywords
        'missing_keywords': [kw for kw, _ in top_missing[:20]],  # Top 20 missing tech keywords
        'missing_keywords_with_freq': top_missing[:20],
        'present_keywords': list(tech_overlap)[:20],  # Only tech keywords
        'jd_keyword_count': len(jd_keywords),  # Tech keywords only
        'resume_keyword_count': len(resume_keywords),  # Tech keywords only
        'overlap_count': tech_overlap_count,
        'tech_overlap_count': tech_overlap_count,
        'tech_total_count': len(jd_keywords)
    }


def get_chunks_text(chunks: List[Dict]) -> str:
    """Extract text from chunks for analysis."""
    return " ".join([chunk.get('text', '') for chunk in chunks])

