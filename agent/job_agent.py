"""
Job Agent - Resume Coach with Multi-Step RAG Pipeline

This agent acts as a resume coach, analyzing job descriptions against resumes using a structured pipeline.

Multi-Step Pipeline:
1. Retrieve top_k chunks from JD and resume separately
2. Extract structured requirements (must-have, nice-to-have, tools, years)
3. Extract resume evidence (projects, tools, quantified impact)
4. Compute match score + gaps
5. Generate recommendations + bullet rewrites

Output Format:
- Structured dict with all extracted data
- Formatted markdown for UI display
"""

from typing import List, Dict, Any, Tuple
from rag.retrieve import FAISSRetriever
from llm.ollama_client import OllamaClient
import re
import json


class JobAgent:
    """RAG agent for answering questions about job descriptions with multi-step pipeline."""
    
    def __init__(self, retriever: FAISSRetriever, llm_client: OllamaClient):
        """
        Initialize agent.
        
        Args:
            retriever: FAISS retriever instance
            llm_client: Ollama LLM client instance
        """
        self.retriever = retriever
        self.llm = llm_client
    
    def analyze_resume_match(self, selected_jd_name: str = None, resume_name: str = "resume.pdf", 
                            top_k: int = 12) -> Dict[str, Any]:
        """
        Multi-step pipeline to analyze resume match against job description.
        
        Args:
            selected_jd_name: Specific JD filename to analyze (e.g., "jd_airbnb.pdf")
            resume_name: Resume filename (default: "resume.pdf")
            top_k: Number of chunks to retrieve per context type
            
        Returns:
            Dict with:
            - structured_data: Complete structured analysis
            - markdown_report: Formatted markdown for UI
            - sources: List of source chunks used
        """
        try:
            # Step 1: Retrieve chunks separately
            jd_chunks, resume_chunks = self._step1_retrieve_chunks(
                selected_jd_name, resume_name, top_k
            )
            
            print(f"[DEBUG] After retrieval: JD chunks={len(jd_chunks) if jd_chunks else 0}, Resume chunks={len(resume_chunks) if resume_chunks else 0}")
            
            # Always use fallback for now to ensure we get results
            # The multi-step pipeline can be enabled later once LLM JSON parsing is more reliable
            if not jd_chunks or not resume_chunks:
                print("[INFO] No chunks found, using fallback")
                return self._fallback_analysis(jd_chunks, resume_chunks, selected_jd_name, resume_name)
            
            # Use fallback (keyword-based) analysis as primary method for reliable results
            print("[INFO] Using keyword-based analysis for reliable results")
            return self._fallback_analysis(jd_chunks, resume_chunks, selected_jd_name, resume_name)
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal structure
            return {
                'structured_data': {
                    'match_score': 0,
                    'requirements': {},
                    'resume_evidence': {},
                    'gaps': {},
                    'recommendations': []
                },
                'markdown_report': f"## Error\n\nAnalysis failed: {str(e)}",
                'sources': []
            }
    
    def _fallback_analysis(self, jd_chunks: List[Dict], resume_chunks: List[Dict], 
                          jd_name: str, resume_name: str) -> Dict[str, Any]:
        """Fallback to simple keyword-based analysis if multi-step pipeline fails."""
        from analytics.skill_gap import compute_skill_gap, get_chunks_text
        
        try:
            jd_text = get_chunks_text(jd_chunks) if jd_chunks else ""
            resume_text = get_chunks_text(resume_chunks) if resume_chunks else ""
            
            if not jd_text or not resume_text:
                return {
                    'structured_data': {
                        'match_score': 0,
                        'requirements': {},
                        'resume_evidence': {},
                        'gaps': {'must_have_skills_missing': [], 'tools_missing': []},
                        'recommendations': []
                    },
                    'markdown_report': "## Error\n\nNo relevant information found in the indexed documents.",
                    'sources': []
                }
            
            # Use keyword-based analysis - TECH KEYWORDS ARE PRIMARY
            gap_analysis = compute_skill_gap(jd_text, resume_text)
            
            # Match score is 100% based on tech keywords from compute_skill_gap
            # compute_skill_gap already applies aggressive boosting, so use it directly
            match_score = int(gap_analysis.get('overlap_percentage', 0))
            
            # Extract missing keywords - ONLY tech keywords (already filtered by compute_skill_gap)
            missing_keywords_raw = gap_analysis.get('missing_keywords', [])
            
            # STRICT filtering: Only keep tech keywords
            from analytics.skill_gap import extract_tech_keywords
            jd_tech_set = extract_tech_keywords(jd_text)
            resume_tech_set = extract_tech_keywords(resume_text)
            all_valid_tech = jd_tech_set.union(resume_tech_set)
            
            # Filter missing keywords to only include valid tech keywords
            missing_keywords = []
            for kw in missing_keywords_raw:
                kw_lower = kw.lower()
                # Check if it's in the tech keyword set (case-insensitive)
                if any(kw_lower == t.lower() for t in all_valid_tech):
                    missing_keywords.append(kw)
                # Also check if it matches tech patterns directly
                elif any(re.search(pattern.replace('\\b', ''), kw_lower, re.IGNORECASE) for pattern in [
                    r'(python|java|javascript|typescript|go|rust|c\+\+|c#|php|ruby|swift|kotlin|scala|r|matlab|perl|bash|shell)',
                    r'(react|angular|vue|django|flask|spring|express|node|next|nuxt|laravel|rails|asp\.net|tensorflow|pytorch|pandas|numpy|scikit|keras)',
                    r'(mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb|sqlite|oracle|sql|nosql|postgres)',
                    r'(aws|azure|gcp|docker|kubernetes|terraform|jenkins|gitlab|github|ci/cd|devops|s3|ec2|lambda|cloudformation)',
                    r'(git|jira|confluence|tableau|powerbi|excel|spark|hadoop|kafka|rabbitmq|airflow|databricks|snowflake|redshift)',
                    r'(agile|scrum|kanban|lean|waterfall|tdd|bdd|microservices|rest|graphql|api|ml|ai|machine learning|deep learning)',
                    r'(html|css|json|xml|yaml|toml|linux|unix|windows|macos|ios|android)'
                ]):
                    missing_keywords.append(kw)
            
            missing_keywords = missing_keywords[:15]  # Limit to top 15
            
            # Get matched tech keywords
            matched_keywords = gap_analysis.get('overlap_keywords', [])[:15]  # Top 15 matched tech keywords
            
            # Generate bullet rewrites using LLM
            bullet_rewrites = []
            try:
                if self.llm:
                    # Check connection
                    if not self.llm.check_connection():
                        print("[WARNING] LLM connection check failed - skipping bullet rewrite generation")
                    else:
                        # Get sample resume bullets (extract from resume text)
                        resume_bullets = self._extract_resume_bullets(resume_text)
                        print(f"[DEBUG] Extracted {len(resume_bullets)} resume bullets for rewriting")
                        
                        if resume_bullets:
                            prompt = f"""You are a resume coach. Rewrite resume bullet points to better match this job description.

=== JOB DESCRIPTION ===
{jd_text[:2000]}

=== CURRENT RESUME BULLETS ===
{chr(10).join([f"{i+1}. {bullet}" for i, bullet in enumerate(resume_bullets[:5])])}

Rewrite 5 resume bullets in XYZ format to better match the job description:
- X = What you did (action)
- Y = Measured result (quantified impact)
- Z = How you did it (method/tool)

Format each bullet as a single line starting with a verb. Use phrases and keywords from the job description.

Return ONLY the 5 rewritten bullets, one per line, numbered 1-5. No explanations, no markdown."""
                            
                            print("[DEBUG] Calling LLM to generate bullet rewrites...")
                            llm_response = self.llm.generate(prompt, temperature=0.7)
                            print(f"[DEBUG] LLM response length: {len(llm_response)}")
                            bullet_rewrites = self._parse_bullet_rewrites(llm_response)
                            print(f"[DEBUG] Generated {len(bullet_rewrites)} bullet rewrites: {bullet_rewrites}")
                        else:
                            print("[WARNING] No resume bullets found to rewrite")
                else:
                    print("[WARNING] LLM client is None - cannot generate bullet rewrites")
            except Exception as e:
                print(f"[ERROR] Bullet rewrite generation failed: {e}")
                import traceback
                traceback.print_exc()
                bullet_rewrites = []
            
            # Format sources
            all_chunks = (jd_chunks or []) + (resume_chunks or [])
            sources = []
            for chunk in all_chunks:
                sources.append({
                    'source': chunk.get('source_file', 'unknown'),
                    'page': chunk.get('page', 0),
                    'chunk_id': chunk.get('chunk_id', 0),
                    'score': chunk.get('score', 0.0),
                    'preview': chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                })
            
            # Build markdown report with ONLY tech keywords
            missing_tech_display = ', '.join(missing_keywords[:10]) if missing_keywords else 'None identified'
            matched_tech_display = ', '.join(matched_keywords[:10]) if matched_keywords else 'None'
            
            return {
                'structured_data': {
                    'match_score': match_score,
                    'requirements': {'must_have_skills': [], 'tools_technologies': []},
                    'resume_evidence': {'key_skills': matched_keywords, 'tools_technologies': matched_keywords},
                    'gaps': {
                        'must_have_skills_missing': missing_keywords,
                        'tools_missing': [],
                        'years_experience_gap': {}
                    },
                    'recommendations': {'bullet_rewrites': bullet_rewrites, 'recommendations': [], 'suggested_projects': []},
                    'bullet_rewrites': bullet_rewrites
                },
                'markdown_report': f"## Resume Match Analysis\n\n**Match Score:** {match_score}/100\n\n**Matched Tech Keywords:** {matched_tech_display}\n\n**Missing Tech Keywords:** {missing_tech_display}",
                'sources': sources,
                'bullet_rewrites': bullet_rewrites,
                'matched_keywords': matched_keywords  # Add matched keywords to result
            }
        except Exception as e:
            print(f"[ERROR] Fallback analysis failed: {e}")
            return {
                'structured_data': {
                    'match_score': 0,
                    'requirements': {},
                    'resume_evidence': {},
                    'gaps': {},
                    'recommendations': []
                },
                'markdown_report': f"## Error\n\nAnalysis failed: {str(e)}",
                'sources': []
            }
    
    def _step1_retrieve_chunks(self, selected_jd_name: str, resume_name: str, top_k: int) -> Tuple[List[Dict], List[Dict]]:
        """Step 1: Retrieve top_k chunks from JD and resume separately."""
        print(f"[DEBUG] Retrieving chunks: JD={selected_jd_name}, Resume={resume_name}, top_k={top_k}")
        
        # Retrieve JD chunks
        if selected_jd_name and selected_jd_name != "All JDs" and selected_jd_name != "All Job Descriptions":
            jd_chunks = self.retriever.retrieve(
                "skills tools technologies responsibilities requirements qualifications experience needed years",
                top_k=top_k,
                include_files=[selected_jd_name]
            )
            # Filter to ensure only from selected JD
            jd_chunks = [chunk for chunk in jd_chunks if chunk.get('source_file', '') == selected_jd_name]
            print(f"[DEBUG] Retrieved {len(jd_chunks)} JD chunks from {selected_jd_name}")
        else:
            jd_chunks = self.retriever.retrieve(
                "skills tools technologies responsibilities requirements qualifications experience needed",
                top_k=top_k,
                exclude_files=[resume_name]
            )
            print(f"[DEBUG] Retrieved {len(jd_chunks)} JD chunks (all JDs)")
        
        # Retrieve Resume chunks
        resume_chunks = self.retriever.retrieve(
            "experience skills projects achievements responsibilities education quantified impact metrics results",
            top_k=top_k,
            include_files=[resume_name]
        )
        # Filter to ensure only from selected resume
        resume_chunks = [chunk for chunk in resume_chunks if chunk.get('source_file', '') == resume_name]
        print(f"[DEBUG] Retrieved {len(resume_chunks)} resume chunks from {resume_name}")
        
        return jd_chunks, resume_chunks
    
    def _step2_extract_requirements(self, jd_chunks: List[Dict], jd_name: str) -> Dict[str, Any]:
        """Step 2: Extract structured requirements (must-have, nice-to-have, tools, years)."""
        jd_text = "\n\n".join([chunk.get('text', '') for chunk in jd_chunks])
        jd_text = jd_text[:3000] if len(jd_text) > 3000 else jd_text
        
        if not jd_text.strip():
            return {
                'must_have_skills': [],
                'nice_to_have_skills': [],
                'tools_technologies': [],
                'years_experience': {'minimum': None, 'preferred': None, 'description': ''},
                'responsibilities': [],
                'qualifications': []
            }
        
        prompt = f"""Extract structured requirements from this job description.

=== JOB DESCRIPTION ===
{jd_text}

Extract and structure the requirements in JSON format:

{{
    "must_have_skills": ["skill1", "skill2", ...],
    "nice_to_have_skills": ["skill1", "skill2", ...],
    "tools_technologies": ["tool1", "tool2", ...],
    "years_experience": {{
        "minimum": number or null,
        "preferred": number or null,
        "description": "text description if no specific number"
    }},
    "responsibilities": ["responsibility1", "responsibility2", ...],
    "qualifications": ["qualification1", "qualification2", ...]
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no explanations
- Be specific and extract actual requirements mentioned
- If years not specified, set to null
- List must-have vs nice-to-have based on language (required, preferred, etc.)"""

        try:
            response = self.llm.generate(prompt, temperature=0.2)
        except Exception as e:
            print(f"[WARNING] LLM call failed in step 2: {e}")
            response = ""
        
        # Parse JSON response
        requirements = self._parse_json_response(response, {
            'must_have_skills': [],
            'nice_to_have_skills': [],
            'tools_technologies': [],
            'years_experience': {'minimum': None, 'preferred': None, 'description': ''},
            'responsibilities': [],
            'qualifications': []
        })
        
        return requirements
    
    def _step3_extract_resume_evidence(self, resume_chunks: List[Dict], resume_name: str) -> Dict[str, Any]:
        """Step 3: Extract resume evidence (projects, tools, quantified impact)."""
        resume_text = "\n\n".join([chunk['text'] for chunk in resume_chunks])
        resume_text = resume_text[:3000] if len(resume_text) > 3000 else resume_text
        
        prompt = f"""Extract structured evidence from this resume.

=== RESUME ===
{resume_text}

Extract and structure the evidence in JSON format:

{{
    "projects": [
        {{
            "name": "project name",
            "technologies": ["tech1", "tech2"],
            "impact": "quantified impact/metrics",
            "description": "brief description"
        }}
    ],
    "tools_technologies": ["tool1", "tool2", ...],
    "quantified_achievements": [
        {{
            "achievement": "what was achieved",
            "metric": "number/percentage",
            "context": "brief context"
        }}
    ],
    "years_experience": number or null,
    "key_skills": ["skill1", "skill2", ...]
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no explanations
- Extract actual numbers, percentages, metrics
- List all technologies and tools mentioned
- Be specific about quantified achievements"""

        try:
            response = self.llm.generate(prompt, temperature=0.2)
        except Exception as e:
            print(f"[WARNING] LLM call failed in step 3: {e}")
            response = ""
        
        # Parse JSON response
        evidence = self._parse_json_response(response, {
            'projects': [],
            'tools_technologies': [],
            'quantified_achievements': [],
            'years_experience': None,
            'key_skills': []
        })
        
        return evidence
    
    def _step4_compute_match(self, requirements: Dict[str, Any], resume_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Compute match score and identify gaps."""
        # Calculate skill matches
        must_have_skills = set(requirements.get('must_have_skills', []))
        nice_to_have_skills = set(requirements.get('nice_to_have_skills', []))
        resume_skills = set(resume_evidence.get('key_skills', []))
        
        must_have_matched = must_have_skills.intersection(resume_skills)
        nice_to_have_matched = nice_to_have_skills.intersection(resume_skills)
        
        # Calculate tool matches
        required_tools = set(requirements.get('tools_technologies', []))
        resume_tools = set(resume_evidence.get('tools_technologies', []))
        tools_matched = required_tools.intersection(resume_tools)
        
        # Calculate gaps
        must_have_gaps = must_have_skills - resume_skills
        nice_to_have_gaps = nice_to_have_skills - resume_skills
        tools_gaps = required_tools - resume_tools
        
        # Calculate match score (weighted)
        total_must_have = len(must_have_skills) if must_have_skills else 1
        total_nice_to_have = len(nice_to_have_skills) if nice_to_have_skills else 1
        total_tools = len(required_tools) if required_tools else 1
        
        must_have_score = (len(must_have_matched) / total_must_have) * 50  # 50% weight
        nice_to_have_score = (len(nice_to_have_matched) / total_nice_to_have) * 20  # 20% weight
        tools_score = (len(tools_matched) / total_tools) * 30  # 30% weight
        
        match_score = int(must_have_score + nice_to_have_score + tools_score)
        match_score = min(100, max(0, match_score))  # Clamp to 0-100
        
        return {
            'match_score': match_score,
            'gaps': {
                'must_have_skills_missing': list(must_have_gaps),
                'nice_to_have_skills_missing': list(nice_to_have_gaps),
                'tools_missing': list(tools_gaps),
                'years_experience_gap': self._check_years_gap(
                    requirements.get('years_experience', {}),
                    resume_evidence.get('years_experience')
                )
            },
            'matches': {
                'must_have_skills_matched': list(must_have_matched),
                'nice_to_have_skills_matched': list(nice_to_have_matched),
                'tools_matched': list(tools_matched)
            }
        }
    
    def _step5_generate_recommendations(self, requirements: Dict[str, Any], resume_evidence: Dict[str, Any],
                                       match_analysis: Dict[str, Any], jd_chunks: List[Dict], 
                                       resume_chunks: List[Dict]) -> Dict[str, Any]:
        """Step 5: Generate recommendations and bullet rewrites."""
        jd_text = "\n\n".join([chunk.get('text', '') for chunk in jd_chunks[:5]])
        resume_text = "\n\n".join([chunk.get('text', '') for chunk in resume_chunks[:5]])
        
        gaps_summary = match_analysis['gaps']
        
        prompt = f"""You are a resume coach. Generate actionable recommendations and bullet rewrites.

=== JOB REQUIREMENTS ===
Must-Have Skills: {', '.join(requirements.get('must_have_skills', [])[:10])}
Tools/Technologies: {', '.join(requirements.get('tools_technologies', [])[:10])}
Missing Skills: {', '.join(gaps_summary.get('must_have_skills_missing', [])[:10])}
Missing Tools: {', '.join(gaps_summary.get('tools_missing', [])[:10])}

=== RESUME EVIDENCE ===
{resume_text[:1500]}

Generate recommendations in JSON format:

{{
    "bullet_rewrites": [
        {{
            "original": "original bullet point from resume",
            "rewritten": "Did X, measured by Y, by doing Z",
            "reason": "why this rewrite helps"
        }}
    ],
    "recommendations": [
        {{
            "priority": "high/medium/low",
            "category": "skills/tools/experience/projects",
            "suggestion": "specific actionable suggestion",
            "rationale": "why this helps match the JD"
        }}
    ],
    "suggested_projects": [
        {{
            "project_idea": "project name/idea",
            "technologies": ["tech1", "tech2"],
            "aligns_with": "which JD requirement this addresses"
        }}
    ]
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no explanations
- Use XYZ format for bullet rewrites: "Did X, measured by Y, by doing Z"
- Make recommendations specific and actionable
- Focus on addressing the identified gaps"""

        try:
            response = self.llm.generate(prompt, temperature=0.3)
        except Exception as e:
            print(f"[WARNING] LLM call failed in step 5: {e}")
            response = ""
        
        # Parse JSON response
        recommendations = self._parse_json_response(response, {
            'bullet_rewrites': [],
            'recommendations': [],
            'suggested_projects': []
        })
        
        return recommendations
    
    def _extract_resume_bullets(self, resume_text: str) -> List[str]:
        """Extract bullet points from resume text."""
        bullets = []
        # Look for bullet patterns: •, -, *, or numbered lists
        lines = resume_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 15:  # Skip very short lines
                continue
            
            # Check for bullet markers (•, -, *, or common bullet chars)
            if any(line.startswith(marker) for marker in ['•', '-', '*', '▪', '▸', '▹', '▪', '○', '●']):
                bullet = re.sub(r'^[•\-*▪▸▹○●]\s*', '', line).strip()
                if len(bullet) > 15:  # Only meaningful bullets
                    bullets.append(bullet)
            # Check for numbered bullets (1., 2., etc.)
            elif re.match(r'^\d+[\.\)]\s+', line):
                bullet = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                if len(bullet) > 15:
                    bullets.append(bullet)
            # Also look for lines that start with action verbs (common in resume bullets)
            elif re.match(r'^(Developed|Implemented|Created|Built|Designed|Managed|Led|Improved|Increased|Reduced|Optimized|Deployed|Maintained|Collaborated|Designed|Architected)', line, re.IGNORECASE):
                if len(line) > 20:  # Only meaningful bullets
                    bullets.append(line)
        
        # If we didn't find enough bullets, try extracting sentences that look like achievements
        if len(bullets) < 3:
            # Look for sentences with numbers (quantified achievements)
            sentences = re.split(r'[.!?]\s+', resume_text)
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for quantified achievements (numbers, percentages, etc.)
                if re.search(r'\d+%|\d+\s*(years|months|users|customers|projects|team)', sentence, re.IGNORECASE):
                    if len(sentence) > 20 and sentence not in bullets:
                        bullets.append(sentence)
        
        print(f"[DEBUG] Extracted {len(bullets)} resume bullets")
        return bullets[:10]  # Return top 10 bullets
    
    def _parse_bullet_rewrites(self, llm_response: str) -> List[str]:
        """Parse bullet rewrites from LLM response."""
        bullets = []
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering (1., 2., etc.)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            # Remove markdown formatting
            line = re.sub(r'^\*\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            if len(line) > 20:  # Only meaningful bullets
                bullets.append(line)
        
        return bullets[:5]  # Return top 5
    
    def _check_years_gap(self, required_years: Dict[str, Any], resume_years: Any) -> Dict[str, Any]:
        """Check if years of experience meet requirements."""
        gap_info = {
            'meets_requirement': True,
            'gap_description': ''
        }
        
        if not resume_years:
            gap_info['meets_requirement'] = False
            gap_info['gap_description'] = 'Years of experience not specified in resume'
            return gap_info
        
        min_required = required_years.get('minimum')
        preferred = required_years.get('preferred')
        
        if min_required and resume_years < min_required:
            gap_info['meets_requirement'] = False
            gap_info['gap_description'] = f'Resume shows {resume_years} years, but JD requires minimum {min_required} years'
        elif preferred and resume_years < preferred:
            gap_info['meets_requirement'] = True
            gap_info['gap_description'] = f'Resume shows {resume_years} years, but JD prefers {preferred} years'
        else:
            gap_info['meets_requirement'] = True
            gap_info['gap_description'] = 'Years of experience meet requirements'
        
        return gap_info
    
    def _parse_json_response(self, response: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from LLM response, with fallback to default."""
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                # Merge with default to ensure all keys exist
                result = default.copy()
                result.update(parsed)
                return result
            except json.JSONDecodeError:
                pass
        
        return default
    
    def _generate_markdown_report(self, structured_data: Dict[str, Any], jd_name: str, resume_name: str) -> str:
        """Generate formatted markdown report for UI."""
        match_score = structured_data['match_score']
        requirements = structured_data['requirements']
        gaps = structured_data['gaps']
        recommendations = structured_data['recommendations']
        
        report = f"""# Resume Match Analysis Report

**Job Description:** {jd_name}  
**Resume:** {resume_name}

## 📊 Match Score: {match_score}/100

"""
        
        # Match score interpretation
        if match_score >= 80:
            report += "🎉 **Excellent match!** Your resume strongly aligns with the job requirements.\n\n"
        elif match_score >= 60:
            report += "✅ **Good match** with some areas for improvement.\n\n"
        elif match_score >= 40:
            report += "⚠️ **Moderate match** - significant improvements needed.\n\n"
        else:
            report += "❌ **Low match** - major gaps need to be addressed.\n\n"
        
        # Requirements summary
        report += "## 📋 Job Requirements Summary\n\n"
        if requirements.get('must_have_skills'):
            report += f"**Must-Have Skills:** {', '.join(requirements['must_have_skills'][:10])}\n\n"
        if requirements.get('tools_technologies'):
            report += f"**Tools/Technologies:** {', '.join(requirements['tools_technologies'][:10])}\n\n"
        
        # Gaps
        report += "## 🔍 Identified Gaps\n\n"
        if gaps.get('must_have_skills_missing'):
            report += f"**Missing Must-Have Skills:** {', '.join(gaps['must_have_skills_missing'][:10])}\n\n"
        if gaps.get('tools_missing'):
            report += f"**Missing Tools:** {', '.join(gaps['tools_missing'][:10])}\n\n"
        
        # Recommendations
        if recommendations.get('recommendations'):
            report += "## 💡 Recommendations\n\n"
            for i, rec in enumerate(recommendations['recommendations'][:10], 1):
                priority_emoji = "🔴" if rec.get('priority') == 'high' else "🟡" if rec.get('priority') == 'medium' else "🟢"
                report += f"{priority_emoji} **{i}. {rec.get('suggestion', 'N/A')}**\n"
                report += f"   *{rec.get('rationale', '')}*\n\n"
        
        # Bullet rewrites
        if recommendations.get('bullet_rewrites'):
            report += "## ✏️ Suggested Bullet Rewrites (XYZ Format)\n\n"
            for i, bullet in enumerate(recommendations['bullet_rewrites'][:5], 1):
                report += f"**{i}. {bullet.get('rewritten', 'N/A')}**\n"
                report += f"   *Original:* {bullet.get('original', 'N/A')}\n"
                report += f"   *Why:* {bullet.get('reason', 'N/A')}\n\n"
        
        return report
    
    # Legacy method for backward compatibility
    def answer_question(self, question: str = "Analyze resume match", top_k: int = 10, 
                       selected_jd_name: str = None, selected_resume_name: str = "resume.pdf") -> Dict[str, Any]:
        """
        Legacy method - calls the new multi-step pipeline with fallback.
        Maintains backward compatibility with existing code.
        """
        try:
            result = self.analyze_resume_match(selected_jd_name, selected_resume_name, top_k)
        except Exception as e:
            print(f"[ERROR] analyze_resume_match failed: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal result
            return {
                'match_score': 0,
                'missing_keywords': [],
                'bullet_rewrites': [],
                'suggested_projects': [],
                'answer': f"Error: {str(e)}",
                'sources': []
            }
        
        # Convert to old format
        structured = result.get('structured_data', {})
        
        # Handle bullet_rewrites - check both top-level and structured_data
        bullet_rewrites = result.get('bullet_rewrites', []) or structured.get('bullet_rewrites', [])
        if bullet_rewrites:
            if isinstance(bullet_rewrites[0], dict):
                bullet_rewrites = [b.get('rewritten', '') if isinstance(b, dict) else str(b) for b in bullet_rewrites]
            else:
                bullet_rewrites = [str(b) for b in bullet_rewrites]
        else:
            bullet_rewrites = []
        
        # Handle suggested_projects - could be list of strings or list of dicts
        recommendations = structured.get('recommendations', {})
        suggested_projects = []
        if isinstance(recommendations, dict):
            suggested_projects = recommendations.get('suggested_projects', [])
            if suggested_projects:
                if isinstance(suggested_projects[0], dict):
                    suggested_projects = [p.get('project_idea', '') if isinstance(p, dict) else str(p) for p in suggested_projects]
                else:
                    suggested_projects = [str(p) for p in suggested_projects]
        
        # Get missing keywords - ONLY tech keywords
        gaps = structured.get('gaps', {})
        missing_keywords = []
        if isinstance(gaps, dict):
            missing_keywords = gaps.get('must_have_skills_missing', []) + gaps.get('tools_missing', [])
        
        # STRICT filtering: Ensure only tech keywords are returned
        tech_patterns_lower = [
            'python', 'java', 'javascript', 'typescript', 'go', 'rust', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'bash', 'shell',
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node', 'next', 'nuxt', 'laravel', 'rails', 'asp.net', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit', 'keras',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'sqlite', 'oracle', 'sql', 'nosql', 'postgres',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github', 'ci/cd', 'devops', 's3', 'ec2', 'lambda', 'cloudformation',
            'git', 'jira', 'confluence', 'tableau', 'powerbi', 'excel', 'spark', 'hadoop', 'kafka', 'rabbitmq', 'airflow', 'databricks', 'snowflake', 'redshift',
            'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'tdd', 'bdd', 'microservices', 'rest', 'graphql', 'api', 'ml', 'ai', 'machine learning', 'deep learning',
            'html', 'css', 'json', 'xml', 'yaml', 'toml', 'linux', 'unix', 'windows', 'macos', 'ios', 'android'
        ]
        
        # Filter missing keywords to only include tech keywords
        missing_keywords_filtered = []
        for kw in missing_keywords:
            kw_lower = kw.lower()
            # Check if it matches tech patterns
            if any(tech in kw_lower or kw_lower in tech for tech in tech_patterns_lower):
                missing_keywords_filtered.append(kw)
            # Also check if it's a common tech acronym (2-4 uppercase letters)
            elif re.match(r'^[A-Z]{2,4}$', kw) and kw not in ['US', 'JD', 'PDF', 'URL', 'HTTP', 'HTTPS']:
                missing_keywords_filtered.append(kw)
        
        missing_keywords = missing_keywords_filtered
        
        # Get matched keywords from result
        matched_keywords = result.get('matched_keywords', []) or structured.get('resume_evidence', {}).get('key_skills', [])
        
        return {
            'match_score': structured.get('match_score', 0),
            'missing_keywords': missing_keywords,  # Only tech keywords
            'matched_keywords': matched_keywords,  # Matched tech keywords
            'bullet_rewrites': bullet_rewrites,
            'suggested_projects': suggested_projects,
            'answer': result.get('markdown_report', ''),
            'sources': result.get('sources', [])
        }


if __name__ == "__main__":
    # Test agent
    from llm.ollama_client import OllamaClient
    from rag.retrieve import FAISSRetriever
    
    print("Initializing agent...")
    retriever = FAISSRetriever()
    llm = OllamaClient()
    agent = JobAgent(retriever, llm)
    
    print("\nRunning multi-step analysis...")
    result = agent.analyze_resume_match("jd_airbnb.pdf", "resume.pdf", top_k=10)
    
    print(f"\nMatch Score: {result['structured_data']['match_score']}/100")
    print(f"\nMarkdown Report:\n{result['markdown_report']}")
