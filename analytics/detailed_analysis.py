"""
Detailed Analysis - Extract skills, generate match score, bullet rewrites using LLM
"""

from typing import List, Dict, Any, Tuple
from rag.retrieve import FAISSRetriever
from llm.ollama_client import OllamaClient
from analytics.analysis_cache import get_cached_analysis, save_analysis
import re


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills/keywords from text using simple extraction."""
    # Common skill patterns
    skills = []
    
    # Look for common skill indicators
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:experience|proficiency|expertise|knowledge|skills?)\b',
        r'\b(?:proficient|experienced|skilled|expert)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+developer|engineer|specialist',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    
    return list(set(skills))


def analyze_jd_resume_match(
    retriever: FAISSRetriever,
    llm: OllamaClient,
    jd_name: str,
    resume_name: str = "resume.pdf"
) -> Dict[str, Any]:
    """
    Perform detailed analysis of JD vs Resume match.
    
    Returns:
        Dict with:
        - match_score: int (0-100)
        - matched_skills: List[str] (top 15)
        - missing_skills: List[str] (top 15)
        - bullet_rewrites: List[str] (5 bullets in XYZ format)
    """
    # Check cache first
    cached = get_cached_analysis(jd_name, resume_name)
    if cached and cached.get('analysis'):
        print(f"[Cache Hit] Using cached analysis for {jd_name} vs {resume_name}")
        return cached.get('analysis', {})
    
    print(f"[Cache Miss] Running fresh analysis for {jd_name} vs {resume_name}")
    
    # Retrieve JD context - STRICTLY from selected JD only
    jd_chunks = retriever.retrieve(
        "skills tools technologies responsibilities requirements qualifications experience needed",
        top_k=15,  # Get more chunks to ensure we have enough
        include_files=[jd_name] if jd_name else None,
        exclude_files=[resume_name] if jd_name else [resume_name]
    )
    
    # Filter to ensure we only have chunks from the selected JD
    if jd_name:
        jd_chunks = [chunk for chunk in jd_chunks if chunk.get('source_file', '') == jd_name]
        print(f"[JD Retrieval] Found {len(jd_chunks)} chunks from {jd_name}")
    
    # Retrieve Resume context - STRICTLY from selected resume only
    resume_chunks = retriever.retrieve(
        "experience skills projects achievements responsibilities education",
        top_k=15,  # Get more chunks
        include_files=[resume_name]
    )
    
    # Filter to ensure we only have chunks from the selected resume
    if resume_name:
        resume_chunks = [chunk for chunk in resume_chunks if chunk.get('source_file', '') == resume_name]
        print(f"[Resume Retrieval] Found {len(resume_chunks)} chunks from {resume_name}")
    
    if not jd_chunks:
        print(f"[ERROR] No chunks found for JD: {jd_name}")
        return {
            'match_score': 0,
            'matched_skills': [],
            'missing_skills': [f"Error: No content found for {jd_name}"],
            'bullet_rewrites': []
        }
    
    if not resume_chunks:
        print(f"[ERROR] No chunks found for Resume: {resume_name}")
        return {
            'match_score': 0,
            'matched_skills': [],
            'missing_skills': [f"Error: No content found for {resume_name}"],
            'bullet_rewrites': []
        }
    
    # Build contexts
    jd_text = "\n\n".join([chunk['text'] for chunk in jd_chunks])
    resume_text = "\n\n".join([chunk['text'] for chunk in resume_chunks])
    
    # Limit context for faster processing
    jd_text = jd_text[:2500] if len(jd_text) > 2500 else jd_text
    resume_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
    
    # Create structured prompt for LLM
    prompt = f"""You are a resume analysis expert. Analyze the job description and resume to provide structured output.

=== JOB DESCRIPTION ({jd_name}) ===
{jd_text}

=== RESUME ===
{resume_text}

Provide analysis in this EXACT format:

MATCH SCORE: [number 0-100]

MATCHED SKILLS (top 15, most relevant):
- [skill1]
- [skill2]
- ...

MISSING SKILLS (top 15, most important):
- [skill1] (why it's needed: [brief reason])
- [skill2] (why it's needed: [brief reason])
- ...

BULLET REWRITES (5 bullets, XYZ format using JD phrases):
- Did [X - action/achievement], measured by [Y - metric/result], by doing [Z - method/approach] (use phrases from JD)
- Did [X], measured by [Y], by doing [Z]
- Did [X], measured by [Y], by doing [Z]
- Did [X], measured by [Y], by doing [Z]
- Did [X], measured by [Y], by doing [Z]

IMPORTANT:
- Start with "MATCH SCORE: " followed by a number 0-100
- List exactly 15 matched skills (or fewer if not available)
- List exactly 15 missing skills (or fewer if not available)
- Provide exactly 5 bullet rewrites in XYZ format
- Use specific phrases and terminology from the job description in bullet rewrites
- Be concise and actionable"""

    # Generate analysis
    response = llm.generate(prompt, temperature=0.3)
    
    # Parse response
    analysis = parse_analysis_response(response)
    
    # Cache the result
    save_analysis(jd_name, resume_name, analysis)
    
    return analysis


def parse_analysis_response(response: str) -> Dict[str, Any]:
    """Parse structured LLM response into analysis dict."""
    result = {
        'match_score': 0,
        'matched_skills': [],
        'missing_skills': [],
        'bullet_rewrites': []
    }
    
    # Extract match score
    match = re.search(r'MATCH SCORE:\s*(\d+)', response, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            result['match_score'] = score
    
    # Extract matched skills
    match = re.search(r'MATCHED SKILLS.*?:\s*\n(.*?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                skill = line[1:].strip()
                if skill:
                    result['matched_skills'].append(skill)
    
    # Extract missing skills
    match = re.search(r'MISSING SKILLS.*?:\s*\n(.*?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                skill = line[1:].strip()
                if skill:
                    result['missing_skills'].append(skill)
    
    # Extract bullet rewrites
    match = re.search(r'BULLET REWRITES.*?:\s*\n(.*?)(?=\n[A-Z]+:|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                bullet = line[1:].strip()
                if bullet:
                    result['bullet_rewrites'].append(bullet)
    
    # Limit to requested counts
    result['matched_skills'] = result['matched_skills'][:15]
    result['missing_skills'] = result['missing_skills'][:15]
    result['bullet_rewrites'] = result['bullet_rewrites'][:5]
    
    return result

