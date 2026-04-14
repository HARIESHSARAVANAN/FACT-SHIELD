import re
import os
import hashlib
import streamlit as st
import requests
import spacy
from datetime import datetime
from groq import Groq
from textblob import TextBlob
from transformers import pipeline
from typing import Dict, Tuple, List, Optional, Any
import pickle
from pathlib import Path
import torch
import json
import time
from functools import wraps

# ====================== CONFIGURATION ======================
CACHE_FILE = "fact_check_cache.json"
MAX_CACHE_SIZE = 1000
DEFAULT_MODEL = "llama-3.1-8b-instant" 

# 🔒 SECURITY NOTE: Consider using environment variables for production
GROQ_API_KEY = "API KEY"   

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_models():
    """Load NLP models with error handling"""
    try:
        # Try to load the model, download if not available
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download the model if not available
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        
        # Load bias classifier
        try:
            bias_classifier = pipeline(
                "text-classification", 
                model="unitary/unbiased-toxic-roberta"
            )
        except Exception as e:
            st.error(f"❌ Failed to load bias classifier: {e}")
            # Fallback to a simpler bias detection
            bias_classifier = None
        
        return nlp, bias_classifier
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        # Provide fallback options
        st.info("🔧 Attempting to use fallback text processing...")
        return None, None

# Initialize models globally
nlp, bias_classifier = load_models()

# ====================== UTILITY DECORATORS ======================
def rate_limit(max_calls: int = 10, period: int = 60):
    """Decorator to limit API calls"""
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside the current period
            calls[:] = [call for call in calls if now - call < period]
            
            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    calls.pop(0)
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def safe_api_call(func):
    """Wrapper for API calls with comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            return None, "⏰ API request timed out"
        except requests.exceptions.ConnectionError:
            return None, "🔌 Connection error - check internet"
        except Exception as e:
            return None, f"❌ API error: {str(e)}"
    return wrapper

# ====================== SECURITY CHECKS ======================
def check_api_key_safety():
    """Verify the API key isn't an example or placeholder"""
    unsafe_patterns = [
        "your_api_key_here",
        "example",
        "test",
        "paste_your",
        "insert_your"
    ]
    
    if not GROQ_API_KEY or any(pattern in GROQ_API_KEY.lower() for pattern in unsafe_patterns):
        st.error("""
        🔒 SECURITY ALERT:
        
        1. You're using an example API key
        2. Please replace GROQ_API_KEY in the code
        3. Never commit this file with real keys to version control
        
        Add this file to .gitignore before proceeding!
        """)
        st.stop()

# ====================== ENHANCED CACHE SYSTEM ======================
class EnhancedFactCheckCache:
    def __init__(self, cache_file: str = CACHE_FILE, max_size: int = MAX_CACHE_SIZE):
        self.cache_file = Path(cache_file)
        self.max_size = max_size
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file safely"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if isinstance(cache_data, dict):
                        return cache_data
            return {}
        except Exception as e:
            st.warning(f"Cache loading failed: {e}")
            return {}
    
    def save_cache(self):
        """Save cache with atomic write and size limits"""
        try:
            # Trim cache if too large
            if len(self.cache) > self.max_size:
                self.cache = dict(list(self.cache.items())[-self.max_size:])
            
            # Create backup if exists
            backup_file = None
            if self.cache_file.exists():
                backup_file = self.cache_file.with_suffix('.bak')
                self.cache_file.rename(backup_file)
            
            # Save new cache
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
            
            # Remove backup if successful
            if backup_file and backup_file.exists():
                backup_file.unlink()
                
        except Exception as e:
            st.warning(f"Cache save failed: {e}")
    
    def get(self, key: str) -> Optional[Tuple[bool, str]]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Tuple[bool, str]):
        self.cache[key] = value
        self.save_cache()
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        self.save_cache()

# ====================== ENHANCED TEXT PROCESSING ======================
def get_hash(text: str) -> str:
    """Create consistent hash for text"""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def resolve_pronouns(sentence: str, full_text: str) -> str:
    """Replace pronouns with proper nouns"""
    if nlp is None:
        return sentence  # Fallback if spaCy not available
    
    try:
        doc = nlp(full_text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons and any(pronoun in sentence.lower() for pronoun in ["he ", "she", "they"]):
            name = persons[-1]
            replacements = [
                (r"\bHe\b", name),
                (r"\bhe\b", name),
                (r"\bShe\b", name),
                (r"\bshe\b", name),
                (r"\bThey\b", name),
                (r"\bthey\b", name)
            ]
            for pattern, repl in replacements:
                sentence = re.sub(pattern, repl, sentence)
        return sentence
    except Exception:
        return sentence  # Fallback on error

def correct_typos(text: str) -> Tuple[str, str]:
    """Fix spelling errors while preserving original"""
    if len(text.strip()) < 5:
        return text, text
    try:
        corrected = str(TextBlob(text).correct())
        return corrected, text
    except Exception:
        return text, text

def is_disclaimer(sentence: str) -> bool:
    """Detect AI disclaimer phrases"""
    disclaimers = [
        "as of my knowledge", "training data", "i may not",
        "do not have current info", "model cutoff", "as an ai",
        "language model", "i don't have access", "my knowledge is limited",
        "i cannot", "unable to provide", "no information available"
    ]
    return any(d in sentence.lower() for d in disclaimers)

def enhanced_split_sentences(text: str) -> List[str]:
    """Improved sentence splitting with bullet point handling"""
    if not text.strip():
        return []
    
    # Handle bullet points and numbered lists
    text = re.sub(r'^[\s]*[•\-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Use spaCy if available, otherwise fallback to simple splitting
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = []
            
            for sent in doc.sents:
                clean_sent = sent.text.strip()
                if len(clean_sent) > 10:  # Filter out very short fragments
                    sentences.append(clean_sent)
            
            return sentences
        except Exception:
            pass  # Fall through to simple splitting
    
    # Fallback simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def is_temporal_statement(text: str) -> bool:
    """Identify time-sensitive claims"""
    now = datetime.now().year
    keywords = [
        "will", "expected to", "in future", "by",
        "as of", "currently", str(now), str(now+1), str(now+2),
        "next year", "last year", "recently", "soon"
    ]
    return any(kw.lower() in text.lower() for kw in keywords)

# ====================== ENHANCED ANALYSIS FUNCTIONS ======================
def detect_bias(text: str) -> str:
    """Identify potentially biased content"""
    if bias_classifier is None:
        # Fallback bias detection
        biased_words = ["always", "never", "everyone knows", "obviously", "clearly"]
        if any(word in text.lower() for word in biased_words):
            return "⚠️ Potentially Biased (fallback detection)"
        return "✅ Neutral (basic check)"
    
    try:
        result = bias_classifier(text)[0]
        label = result["label"].lower()
        score = result["score"]
        
        if "toxic" in label and score > 0.8:
            return f"⚠️ Potentially Biased (confidence: {score:.2f})"
        elif "toxic" in label and score > 0.6:
            return f"⚖️ Slightly Biased (confidence: {score:.2f})"
        return "✅ Neutral"
    except Exception as e:
        return f"⚠️ Bias Check Failed: {str(e)}"

@safe_api_call
def enhanced_web_verification(query: str) -> Tuple[bool, str]:
    """Improved web verification with multiple sources"""
    sources = [
        f"https://www.google.com/search?q={requests.utils.quote(query)}",
        f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
    ]
    
    verification_count = 0
    total_sources = len(sources)
    
    for source in sources:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            response = requests.get(source, headers=headers, timeout=15)
            
            if response.status_code == 200:
                text = response.text.lower()
                positive_indicators = [
                    "study shows", "research indicates", "according to", 
                    "scientific consensus", "confirmed by", "verified",
                    "experts say", "research confirms", "studies show"
                ]
                negative_indicators = [
                    "myth", "false", "debunked", "incorrect", "not true",
                    "hoax", "fake", "misinformation"
                ]
                
                pos_count = sum(1 for indicator in positive_indicators if indicator in text)
                neg_count = sum(1 for indicator in negative_indicators if indicator in text)
                
                if pos_count > neg_count:
                    verification_count += 1
                    
        except Exception:
            continue
    
    if verification_count >= total_sources * 0.5:  # 50% consensus
        return True, f"🌐 Corroborated by multiple sources ({verification_count}/{total_sources})"
    elif verification_count > 0:
        return False, f"🌐 Mixed evidence found ({verification_count}/{total_sources} sources)"
    else:
        return False, f"🌐 No supporting evidence found"

@rate_limit(max_calls=5, period=60)
def is_factually_correct(statement: str, model: str = DEFAULT_MODEL) -> Tuple[bool, str]:
    """Core fact-checking function with enhanced prompting"""
    cache = EnhancedFactCheckCache()
    statement_hash = get_hash(statement)
    cached_result = cache.get(statement_hash)
    if cached_result:
        return cached_result

    prompt = f"""
    As a professional fact-checker, analyze this statement for factual accuracy:
    "{statement}"

    Consider:
    1. Scientific consensus and evidence
    2. Historical facts and records
    3. Logical consistency
    4. Potential biases or exaggerations

    Respond strictly in this format:
    Verification: [True/False]
    Explanation: [Brief rationale explaining your reasoning]
    Confidence: [High/Medium/Low]
    Sources: [Mention types of sources that would verify this]

    Be objective and evidence-based in your assessment.
    """

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model.strip(),  # Added .strip() to remove any spaces
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            timeout=45
        )
        result = response.choices[0].message.content.strip()
        
        match = re.search(r"Verification:\s*(True|False)", result, re.I)
        verification = bool(match and match.group(1).lower() == "true")

        cache.set(statement_hash, (verification, result))
        return verification, result
    except Exception as e:
        if "model_decommissioned" in str(e) or "not found" in str(e):
            st.error(f"❌ The selected model '{model}' is not available. Please choose a different model from the sidebar.")
            # Fallback to a known working model
            if model != "llama-3.1-8b-instant":
                st.info("🔄 Trying with llama-3.1-8b-instant model instead...")
                return is_factually_correct(statement, "llama-3.1-8b-instant")
        st.error(f"Fact check failed: {str(e)}")
        return False, f"❌ Verification error: {e}"

# ====================== CONTENT REWRITING FUNCTIONS ======================
@rate_limit(max_calls=3, period=60)
def rewrite_content_without_issues(results: List[dict], original_content: str, model: str = DEFAULT_MODEL) -> str:
    """
    Rewrite content removing hallucinations and bias while preserving verified facts
    """
    try:
        # Separate sentences by category
        verified_sentences = []
        problematic_sentences = []
        biased_sentences = []
        
        for result in results:
            if "✅" in result["status"]:
                verified_sentences.append(result["sentence"])
            elif "❌" in result["status"]:
                problematic_sentences.append({
                    "sentence": result["sentence"],
                    "issue": "hallucination",
                    "explanation": result["explanation"]
                })
            
            if "⚠️" in result["bias"]:
                biased_sentences.append({
                    "sentence": result["sentence"],
                    "bias_type": result["bias"]
                })
        
        # Build context for rewriting
        context = {
            "verified": verified_sentences,
            "hallucinations": problematic_sentences,
            "biased": biased_sentences
        }
        
        # Create rewriting prompt
        prompt = f"""
You are an expert content editor. Your task is to rewrite the following content to remove hallucinations and bias while preserving factual accuracy.

ORIGINAL CONTENT:
{original_content}

ANALYSIS RESULTS:
- Verified factual sentences: {len(verified_sentences)}
- Hallucinated/incorrect sentences: {len(problematic_sentences)}
- Biased sentences: {len(biased_sentences)}

HALLUCINATIONS TO REMOVE/FIX:
{chr(10).join([f"- {item['sentence']} (Issue: {item['explanation'][:100]}...)" for item in problematic_sentences]) if problematic_sentences else "None"}

BIASED CONTENT TO NEUTRALIZE:
{chr(10).join([f"- {item['sentence']}" for item in biased_sentences]) if biased_sentences else "None"}

INSTRUCTIONS:
1. Keep all verified factual information ({len(verified_sentences)} sentences)
2. Remove or correct the {len(problematic_sentences)} hallucinated/incorrect statements
3. Rewrite the {len(biased_sentences)} biased sentences in a neutral, objective tone
4. Maintain the original structure and flow where possible
5. Ensure coherence and readability
6. Do not add new information that wasn't verified
7. Use clear, professional language

REWRITTEN CONTENT (provide only the rewritten text, no explanations):
"""

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Low temperature for consistent, factual output
            max_tokens=2048,
            timeout=60
        )
        
        rewritten_content = response.choices[0].message.content.strip()
        return rewritten_content
        
    except Exception as e:
        st.error(f"❌ Content rewriting failed: {str(e)}")
        return None

def generate_rewrite_report(original: str, rewritten: str, results: List[dict]) -> str:
    """Generate a detailed report of the rewriting process"""
    report = []
    report.append("=" * 60)
    report.append("📝 CONTENT REWRITING REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Statistics
    verified = sum(1 for r in results if "✅" in r["status"])
    hallucinations = sum(1 for r in results if "❌" in r["status"])
    biased = sum(1 for r in results if "⚠️" in r["bias"])
    
    report.append("CHANGES MADE:")
    report.append(f"  ✅ Preserved {verified} verified facts")
    report.append(f"  ❌ Removed/corrected {hallucinations} hallucinations")
    report.append(f"  ⚖️ Neutralized {biased} biased statements")
    report.append("")
    
    report.append("ORIGINAL CONTENT:")
    report.append("-" * 60)
    report.append(original)
    report.append("")
    
    report.append("REWRITTEN CONTENT:")
    report.append("-" * 60)
    report.append(rewritten)
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

# ====================== STREAMLIT UI ======================
def setup_sidebar():
    """Configure sidebar options"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        model = st.selectbox(
            "🧠 AI Model",
            [
                "llama-3.1-8b-instant",         # Fast and reliable
                "llama-3.2-1b-preview",         # Newer model
                "llama-3.2-3b-preview",         # Newer model
                "llama-3.2-90b-vision-preview", # If available
                "mixtral-8x7b-32768",           # Alternative
            ],
            index=0
        )
        
        options = {
            "model": model,
            "manual_input": st.checkbox("✍️ Manual Input Mode", value=True),
            "web_check": st.checkbox("🌐 Enable Web Verification", value=True),
            "truth_check": st.checkbox("📌 Compare with Ground Truth"),
            "typo_check": st.checkbox("🄤 Enable Typo Correction", value=True),
            "bias_check": st.checkbox("⚖️ Enable Bias Detection", value=True),
            "auto_rewrite": st.checkbox("✨ Auto-Rewrite Content", value=True, help="Automatically rewrite content after analysis to remove hallucinations and bias")
        }

        if options["truth_check"]:
            options["truth_text"] = st.text_area("📜 Ground Truth (optional):", "", height=100)
        else:
            options["truth_text"] = ""

        st.markdown("---")
        st.subheader("🔧 Advanced Options")
        
        # Model status
        st.write("**Model Status:**")
        if nlp is not None:
            st.success("✅ spaCy model loaded")
        else:
            st.warning("⚠️ Using fallback text processing")
            
        if bias_classifier is not None:
            st.success("✅ Bias classifier loaded")
        else:
            st.warning("⚠️ Using basic bias detection")
        
        # Cache management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache"):
                cache = EnhancedFactCheckCache()
                cache.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("📊 Cache Stats"):
                cache = EnhancedFactCheckCache()
                st.info(f"Cache size: {len(cache.cache)} entries")
        
        st.markdown("---")
        st.warning("⚠️ Never share this app with API keys included!")
        
    return options

def display_confidence_meter(score: int):
    """Visual confidence meter"""
    st.markdown("### 📊 Confidence Level")
    
    if score >= 80:
        color = "green"
        emoji = "🔒"
        status = "High Confidence"
    elif score >= 60:
        color = "orange"
        emoji = "⚠️"
        status = "Moderate Confidence"
    else:
        color = "red"
        emoji = "🔴"
        status = "Low Confidence"
    
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin: 10px 0;">
        <h3 style="margin: 0; color: {color}; display: flex; align-items: center; gap: 10px;">
            {emoji} {status} - Trust Score: {score}/100
        </h3>
        <div style="background-color: {color}40; height: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="background-color: {color}; height: 100%; width: {score}%; border-radius: 10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_results(sentence: str, analysis: dict):
    """Show analysis for each sentence with enhanced visualization"""
    status_emoji = analysis['status'][0]  # Get the emoji
    status_text = analysis['status'][2:]  # Get the text after emoji
    
    with st.expander(f"{status_emoji} {status_text}: {sentence[:80]}...", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**📝 Sentence:** {sentence}")
            if analysis.get('original') and analysis['original'] != sentence:
                st.markdown(f"*🔤 Original (corrected):* ~~{analysis['original']}~~")
            
        with col2:
            st.markdown(f"**{analysis['bias']}**")
            st.markdown(f"**Score:** {analysis['score']}/10")
        
        st.markdown("---")
        st.markdown(f"**🔍 Analysis:** {analysis['explanation']}")
        
        if analysis.get('web_verification'):
            st.markdown(f"**🌐 Web Check:** {analysis['web_verification']}")

def calculate_trust_score(results: List[dict]) -> int:
    """Compute overall trustworthiness score with enhanced weighting"""
    if not results:
        return 0
        
    max_possible = len(results) * 10
    min_possible = len(results) * -15
    
    weighted_score = 0
    for result in results:
        base_score = result["score"]
        
        # Apply weighting based on factors
        if "✅" in result["status"]:
            weighted_score += base_score * 1.2  # Bonus for verified content
        elif "⚠️" in result["bias"]:
            weighted_score += base_score * 0.8  # Penalty for bias
        else:
            weighted_score += base_score
    
    normalized = 50 + (weighted_score / (max_possible - min_possible)) * 100
    return max(0, min(100, round(normalized)))

def generate_report(results: List[dict]) -> str:
    """Create comprehensive downloadable report"""
    report = []
    report.append("=" * 60)
    report.append("🛡️ FactShield AI Auditor - Analysis Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Sentences Analyzed: {len(results)}")
    report.append("")
    
    for i, result in enumerate(results, 1):
        report.append(f"Sentence #{i}:")
        report.append(f"  Text: {result['sentence']}")
        report.append(f"  Status: {result['status']}")
        report.append(f"  Bias Assessment: {result['bias']}")
        report.append(f"  Score: {result['score']}/10")
        report.append(f"  Analysis: {result['explanation']}")
        if result.get('original') and result['original'] != result['sentence']:
            report.append(f"  Original (corrected): {result['original']}")
        report.append("")
    
    # Summary statistics
    verified = sum(1 for r in results if "✅" in r["status"])
    issues = sum(1 for r in results if "❌" in r["status"])
    warnings = sum(1 for r in results if "⚠️" in r["status"])
    
    report.append("SUMMARY STATISTICS:")
    report.append(f"  ✅ Verified: {verified}/{len(results)}")
    report.append(f"  ⚠️ Warnings: {warnings}/{len(results)}")
    report.append(f"  ❌ Issues: {issues}/{len(results)}")
    report.append(f"  📊 Overall Trust Score: {calculate_trust_score(results)}/100")
    
    return "\n".join(report)

def add_export_options(report_data: str, results: List[dict]):
    """Enhanced export options"""
    st.markdown("### 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download TXT Report",
            report_data,
            file_name=f"factshield_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            help="Complete analysis report in text format"
        )
    
    with col2:
        # JSON export
        json_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_sentences": len(results),
                "trust_score": calculate_trust_score(results)
            },
            "results": results
        }
        st.download_button(
            "📊 Download JSON",
            json.dumps(json_data, indent=2),
            file_name=f"factshield_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            help="Structured data for further analysis"
        )
    
    with col3:
        if st.button("📋 Copy to Clipboard"):
            st.code(report_data[:1000] + "..." if len(report_data) > 1000 else report_data)
            st.success("Report copied to clipboard!")

def analyze_content(content: str, options: dict):
    """Main analysis workflow with enhanced features"""
    st.subheader("🔍 Analysis Results")
    
    # Pre-process content
    clean_content = re.sub(r'^[\s]*[•\-*]\s+', '', content, flags=re.MULTILINE)
    
    with st.spinner("🔬 Processing content... This may take a few moments."):
        try:
            sentences = enhanced_split_sentences(clean_content)
            
            if not sentences:
                st.warning("No meaningful sentences found to analyze.")
                return
            
            st.info(f"📄 Analyzing {len(sentences)} sentences...")
            
            results = []
            score_breakdown = {
                "verified": 0,
                "disclaimers": 0,
                "minor_issues": 0,
                "hallucinations": 0,
                "biased_content": 0,
                "temporal_claims": 0
            }

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, sentence in enumerate(sentences):
                progress = (i + 1) / len(sentences)
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing sentence {i+1} of {len(sentences)}: {sentence[:50]}...")
                
                # Text processing
                resolved = resolve_pronouns(sentence, content)
                if options["typo_check"]:
                    processed, original = correct_typos(resolved)
                else:
                    processed, original = resolved, resolved

                # Analysis
                temporal = is_temporal_statement(processed)
                valid, explanation = is_factually_correct(processed, options["model"])
                
                if options["bias_check"]:
                    bias = detect_bias(processed)
                else:
                    bias = "⚖️ Bias check disabled"
                
                disclaimer = is_disclaimer(processed)
                
                # Web verification if enabled
                web_verification_msg = ""
                if options["web_check"]:
                    web_valid, web_msg = enhanced_web_verification(processed)
                    explanation += f"\n{web_msg}"
                    web_verification_msg = web_msg
                    if web_valid:
                        valid = True

                # Determine status and score
                if disclaimer and valid:
                    status = "✅ Verified (with Disclaimer)"
                    score = 4
                    score_breakdown["disclaimers"] += 1
                elif disclaimer:
                    status = "⚠️ Disclaimer"
                    score = 2
                    score_breakdown["disclaimers"] += 1
                elif temporal and not valid:
                    status = "🕒 Unverifiable (Future Claim)"
                    explanation += "\n🕒 Time-sensitive claims require ongoing verification"
                    score = 1
                    score_breakdown["temporal_claims"] += 1
                elif temporal and valid:
                    status = "✅ Verified (Temporal)"
                    score = 4
                    score_breakdown["temporal_claims"] += 1
                elif valid:
                    status = "✅ Verified"
                    score = 5
                    score_breakdown["verified"] += 1
                else:
                    status = "❌ Potential Hallucination"
                    score = -7
                    score_breakdown["hallucinations"] += 1

                # Bias penalty
                if "⚠️" in bias and options["bias_check"]:
                    score -= 2
                    score_breakdown["biased_content"] += 1

                results.append({
                    "sentence": processed,
                    "status": status,
                    "bias": bias,
                    "explanation": explanation,
                    "score": score,
                    "original": original if original != processed else None,
                    "web_verification": web_verification_msg
                })

            progress_bar.empty()
            status_text.empty()

            # Display results
            st.subheader("📋 Detailed Analysis")
            for result in results:
                display_results(result["sentence"], result)

            # Overall assessment
            trust_score = calculate_trust_score(results)
            display_confidence_meter(trust_score)
            
            # Detailed breakdown
            with st.expander("📈 Detailed Breakdown", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Verified", score_breakdown['verified'])
                    st.metric("⚠️ Disclaimers", score_breakdown['disclaimers'])
                
                with col2:
                    st.metric("❌ Hallucinations", score_breakdown['hallucinations'])
                    st.metric("🕒 Temporal Claims", score_breakdown['temporal_claims'])
                
                with col3:
                    st.metric("⚖️ Biased Content", score_breakdown['biased_content'])
                    st.metric("📊 Total Sentences", len(sentences))

            # Export options for original analysis
            report = generate_report(results)
            add_export_options(report, results)
            
            # ====================== NEW CONTENT REWRITING SECTION ======================
            st.markdown("---")
            st.markdown("## ✨ Content Rewriting")
            
            # Check if there are issues to fix
            has_hallucinations = score_breakdown['hallucinations'] > 0
            has_bias = score_breakdown['biased_content'] > 0
            
            if has_hallucinations or has_bias:
                st.warning(f"⚠️ Found {score_breakdown['hallucinations']} hallucination(s) and {score_breakdown['biased_content']} biased statement(s)")
                
                # Show rewrite button or auto-rewrite
                if options.get("auto_rewrite", False):
                    rewrite_triggered = True
                    st.info("🔄 Auto-rewriting enabled. Generating improved content...")
                else:
                    rewrite_triggered = st.button("✍️ Rewrite Content (Remove Hallucinations & Bias)", type="primary")
                
                if rewrite_triggered:
                    with st.spinner("✨ Rewriting content to remove issues... This may take a moment."):
                        rewritten_content = rewrite_content_without_issues(results, content, options["model"])
                        
                        if rewritten_content:
                            st.success("✅ Content successfully rewritten!")
                            
                            # Display comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 📄 Original Content")
                                st.markdown(f"""
                                <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 5px solid #f44336;">
                                    {content.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                **Issues Found:**
                                - ❌ {score_breakdown['hallucinations']} Hallucination(s)
                                - ⚖️ {score_breakdown['biased_content']} Biased Statement(s)
                                - 📊 Trust Score: {trust_score}/100
                                """)
                            
                            with col2:
                                st.markdown("### ✨ Rewritten Content")
                                st.markdown(f"""
                                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4caf50;">
                                    {rewritten_content.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                **Improvements Made:**
                                - ✅ {score_breakdown['verified']} Facts Preserved
                                - 🔧 {score_breakdown['hallucinations']} Issue(s) Fixed
                                - ⚖️ {score_breakdown['biased_content']} Statement(s) Neutralized
                                """)
                            
                            # Show rewritten content in expandable section
                            with st.expander("📝 View Full Rewritten Content", expanded=False):
                                st.text_area("Rewritten Content:", rewritten_content, height=300)
                            
                            # Export options for rewritten content
                            st.markdown("### 📤 Export Rewritten Content")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.download_button(
                                    "📥 Download Rewritten Content",
                                    rewritten_content,
                                    file_name=f"rewritten_content_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    help="Download the improved content"
                                )
                            
                            with col2:
                                # Generate comprehensive rewrite report
                                rewrite_report = generate_rewrite_report(content, rewritten_content, results)
                                st.download_button(
                                    "📊 Download Rewrite Report",
                                    rewrite_report,
                                    file_name=f"rewrite_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    help="Detailed report of changes made"
                                )
                            
                            with col3:
                                # JSON export with both versions
                                comparison_data = {
                                    "metadata": {
                                        "generated": datetime.now().isoformat(),
                                        "original_trust_score": trust_score,
                                        "hallucinations_fixed": score_breakdown['hallucinations'],
                                        "bias_neutralized": score_breakdown['biased_content'],
                                        "facts_preserved": score_breakdown['verified']
                                    },
                                    "original_content": content,
                                    "rewritten_content": rewritten_content,
                                    "analysis_results": results
                                }
                                st.download_button(
                                    "📋 Download Comparison JSON",
                                    json.dumps(comparison_data, indent=2),
                                    file_name=f"content_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                    help="Both versions with analysis data"
                                )
                            
                            # Highlight key changes
                            st.markdown("---")
                            st.markdown("### 🔍 Key Changes Made")
                            
                            with st.expander("View Detailed Change Summary", expanded=False):
                                st.markdown("#### ❌ Removed/Corrected (Hallucinations):")
                                hallucinated_items = [r for r in results if "❌" in r["status"]]
                                if hallucinated_items:
                                    for item in hallucinated_items:
                                        st.markdown(f"- ~~{item['sentence']}~~")
                                        st.caption(f"   Reason: {item['explanation'][:150]}...")
                                else:
                                    st.info("No hallucinations found")
                                
                                st.markdown("#### ⚖️ Neutralized (Bias):")
                                biased_items = [r for r in results if "⚠️" in r["bias"]]
                                if biased_items:
                                    for item in biased_items:
                                        st.markdown(f"- {item['sentence']}")
                                        st.caption(f"   Bias Type: {item['bias']}")
                                else:
                                    st.info("No biased content found")
                                
                                st.markdown("#### ✅ Preserved (Verified Facts):")
                                verified_items = [r for r in results if "✅" in r["status"]]
                                if verified_items:
                                    for item in verified_items[:5]:  # Show first 5
                                        st.markdown(f"- {item['sentence']}")
                                    if len(verified_items) > 5:
                                        st.caption(f"   ... and {len(verified_items) - 5} more verified statements")
                                else:
                                    st.info("No verified facts found")
                        else:
                            st.error("❌ Failed to rewrite content. Please try again.")
            else:
                st.success("✅ No hallucinations or bias detected! Your content is already clean and trustworthy.")
                st.info("🎉 Trust Score: " + str(trust_score) + "/100 - No rewriting needed!")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

# ====================== MAIN APPLICATION ======================
def main():
    """Application entry point"""
    check_api_key_safety()
    st.set_page_config(
        page_title="🛡️ FactShield AI Auditor", 
        layout="wide",
        page_icon="🛡️"
    )
    
    st.title("🛡️ FactShield AI Auditor")
    st.markdown("### Analyze AI-generated content for accuracy, bias, and reliability")
    
    # Show model status
    if nlp is None or bias_classifier is None:
        st.warning("⚠️ Some models failed to load. Using fallback functionality. The app will still work but with reduced accuracy.")
    
    # Header with information
    with st.expander("ℹ️ About FactShield", expanded=False):
        st.markdown("""
        **FactShield** helps you verify AI-generated content by:
        - ✅ **Fact-checking** claims against known information
        - ⚖️ **Detecting bias** in the content
        - 🌐 **Web verification** against online sources
        - 📊 **Scoring trustworthiness** of the entire content
        - ✨ **Rewriting content** to remove hallucinations and bias
        
        **How to use:**
        1. Choose your analysis mode (Manual Input or Generate & Analyze)
        2. Configure settings in the sidebar
        3. Run the analysis
        4. Review the detailed results and trust score
        5. Let FactShield automatically rewrite the content to fix issues
        6. Download reports and improved content
        """)

    options = setup_sidebar()
    
    if options["manual_input"]:
        st.subheader("✍️ Manual Content Analysis")
        content = st.text_area(
            "Paste AI-generated content to analyze:",
            height=300,
            placeholder="Paste the content you want to analyze here..."
        )
        if st.button("🔍 Analyze Content", type="primary") and content.strip():
            analyze_content(content, options)
        elif content.strip():
            st.info("💡 Click the 'Analyze Content' button to start analysis")
    else:
        st.subheader("🤖 Generate & Analyze")
        prompt = st.text_area(
            "Enter your prompt for analysis:",
            height=150,
            placeholder="Enter a prompt to generate and analyze content..."
        )
        if st.button("🚀 Generate & Analyze", type="primary") and prompt.strip():
            with st.spinner("🤖 Generating response..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model=options["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1024,
                        timeout=30
                    )
                    content = response.choices[0].message.content
                    
                    st.subheader("📄 Generated Content")
                    st.write(content)
                    
                    analyze_content(content, options)
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
        elif prompt.strip():
            st.info("💡 Click the 'Generate & Analyze' button to create and analyze content")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🛡️ FactShield AI Auditor v2.0 | Now with Content Rewriting | Use responsibly | Keep your API keys secure"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
