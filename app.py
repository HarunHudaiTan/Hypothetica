"""
Hypothetica - Research Originality Assessment System
Streamlit UI with real-time progress updates.
"""
import streamlit as st
import logging
from typing import List, Dict

from pipeline.originality_pipeline import OriginalityPipeline
from models.analysis import OriginalityLabel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hypothetica - Research Originality",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* Main container - removed dark gradient for better compatibility */

/* Headers */
h1 {
    color: #1a1a2e !important;
    font-weight: 700;
}

/* Card styling */
.stCard {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Sentence highlighting - light theme compatible */
.sentence-high {
    background: #dcfce7 !important;
    border-left: 4px solid #22c55e;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #166534;
}
.sentence-high:hover {
    background: #bbf7d0 !important;
}

.sentence-medium {
    background: #fef9c3 !important;
    border-left: 4px solid #eab308;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #854d0e;
}
.sentence-medium:hover {
    background: #fef08a !important;
}

.sentence-low {
    background: #fee2e2 !important;
    border-left: 4px solid #ef4444;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #991b1b;
}
.sentence-low:hover {
    background: #fecaca !important;
}

/* Gauge meter container */
.gauge-container {
    text-align: center;
    padding: 20px;
}

/* Match card */
.match-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #6366f1) !important;
}

/* Text areas - ensure visible text on light theme */
.stTextArea textarea {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    font-size: 16px !important;
    padding: 12px !important;
}

.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
}

/* Text input fields */
.stTextInput input {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* Metric styling */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Legend */
.legend {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 16px 0;
    padding: 12px;
    background: #f8fafc;
    border-radius: 8px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    color: #334155;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'step': 'input',  # input, questions, processing, results
        'pipeline': None,
        'user_idea': '',
        'followup_questions': [],
        'followup_answers': [],
        'result': None,
        'selected_sentence_idx': None,
        'progress_message': '',
        'progress_pct': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    """Render the header section."""
    st.markdown("""
    <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 8px; color: white !important;">🔬 Hypothetica</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0;">
            AI-Powered Research Originality Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_gauge(score: int):
    """Render the originality gauge meter using Streamlit native components."""
    # Determine color and label based on score
    if score >= 70:
        emoji = "🟢"
        label = "High Originality"
        hex_color = "#22c55e"
    elif score >= 40:
        emoji = "🟡"
        label = "Moderate Originality"
        hex_color = "#eab308"
    else:
        emoji = "🔴"
        label = "Low Originality"
        hex_color = "#ef4444"
    
    # Use columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center; font-size: 5rem; margin: 0;'>{emoji}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 4rem; margin: 0; color: {hex_color};'>{score}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin: 0;'>/ 100</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.3rem; font-weight: 600; color: {hex_color}; margin-top: 10px;'>{label}</p>", unsafe_allow_html=True)
    
    # Progress bar as visual gauge
    st.progress(score / 100)


def render_sentence_with_highlighting(annotations):
    """Render sentences with color-coded highlighting."""
    st.markdown("""
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: #22c55e;"></div>
            <span>High Originality</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #eab308;"></div>
            <span>Moderate</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #ef4444;"></div>
            <span>Low Originality</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for ann in annotations:
        label_class = {
            OriginalityLabel.HIGH: "sentence-high",
            OriginalityLabel.MEDIUM: "sentence-medium",
            OriginalityLabel.LOW: "sentence-low"
        }.get(ann.label, "sentence-high")
        
        # Create clickable sentence
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(
                f'<div class="{label_class}">{ann.sentence}</div>',
                unsafe_allow_html=True
            )
        with col2:
            if ann.label != OriginalityLabel.HIGH and ann.linked_sections:
                if st.button("🔍", key=f"sent_{ann.index}", help="View matching sources"):
                    st.session_state.selected_sentence_idx = ann.index


def render_matches_panel(pipeline, sentence_idx, annotations):
    """Render the matching sources panel for a selected sentence."""
    if sentence_idx is None or sentence_idx >= len(annotations):
        return
    
    ann = annotations[sentence_idx]
    
    st.markdown("### 📄 Matching Sources")
    st.markdown(f"**Selected sentence:** *{ann.sentence}*")
    st.markdown(f"**Overlap score:** {ann.overlap_score:.2%}")
    
    # Get matches from RAG
    matches = pipeline.get_matches_for_sentence(ann.sentence, top_k=5)
    
    if not matches and ann.linked_sections:
        # Use linked sections from Layer 1 analysis
        matches = [
            {
                "paper_title": ls.paper_title,
                "heading": ls.heading,
                "text": ls.text_snippet,
                "similarity": ls.similarity,
                "reason": ls.reason
            }
            for ls in ann.linked_sections
        ]
    
    if matches:
        for i, match in enumerate(matches):
            with st.expander(f"📑 {match.get('paper_title', 'Unknown Paper')[:50]}...", expanded=(i == 0)):
                st.markdown(f"**Section:** {match.get('heading', 'N/A')}")
                st.markdown(f"**Similarity:** {match.get('similarity', 0):.2%}")
                st.markdown("---")
                st.markdown(f"*{match.get('text', 'No text available')[:500]}...*")
    else:
        st.info("No detailed matches found for this sentence.")
    
    if st.button("← Back to results"):
        st.session_state.selected_sentence_idx = None
        st.rerun()


def render_cost_breakdown(cost):
    """Render cost breakdown."""
    cost_dict = cost.to_dict()
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("Follow-up", f"${cost_dict['breakdown']['followup']:.4f}")
    with cols[1]:
        st.metric("Keywords", f"${cost_dict['breakdown']['keywords']:.4f}")
    with cols[2]:
        st.metric("Layer 1", f"${cost_dict['breakdown']['layer1']:.4f}")
    with cols[3]:
        st.metric("Layer 2", f"${cost_dict['breakdown']['layer2']:.4f}")
    with cols[4]:
        st.metric("**Total**", f"${cost_dict['estimated_cost_usd']:.4f}")


# =============================================================================
# MAIN APP LOGIC
# =============================================================================
def main():
    init_session_state()
    render_header()
    
    # =========================================================================
    # STEP 1: INPUT
    # =========================================================================
    if st.session_state.step == 'input':
        st.markdown("### 💡 Enter Your Research Idea")
        st.markdown("*Describe your research idea in detail. The more specific, the better the analysis.*")
        
        user_idea = st.text_area(
            "Your research idea:",
            height=200,
            placeholder="Example: I want to develop a multimodal retrieval-augmented generation (RAG) system that can process and reason over both text documents and images simultaneously...",
            key="idea_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Analyze Originality", type="primary", disabled=len(user_idea) < 50):
                st.session_state.user_idea = user_idea
                st.session_state.pipeline = OriginalityPipeline()
                st.session_state.step = 'questions'
                st.rerun()
        
        if len(user_idea) < 50:
            st.caption("Please enter at least 50 characters to continue.")
    
    # =========================================================================
    # STEP 2: FOLLOW-UP QUESTIONS
    # =========================================================================
    elif st.session_state.step == 'questions':
        st.markdown("### 🤔 Clarifying Questions")
        st.markdown("*Please answer these questions to help us better assess your idea's originality.*")
        
        # Generate questions if not already done
        if not st.session_state.followup_questions:
            with st.spinner("Generating questions..."):
                questions = st.session_state.pipeline.generate_followup_questions(
                    st.session_state.user_idea
                )
                st.session_state.followup_questions = questions
                st.rerun()
        
        # Display questions and collect answers
        answers = []
        for q in st.session_state.followup_questions:
            category_emoji = {
                "problem": "🎯",
                "method": "⚙️",
                "novelty": "✨",
                "application": "🌍"
            }.get(q.get('category', ''), "❓")
            
            answer = st.text_area(
                f"{category_emoji} {q.get('question', 'Question')}",
                key=f"q_{q.get('id', 0)}",
                height=100
            )
            answers.append(answer)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Continue →", type="primary"):
                st.session_state.followup_answers = answers
                st.session_state.step = 'processing'
                st.rerun()
        with col2:
            if st.button("Skip questions"):
                st.session_state.followup_answers = ["" for _ in st.session_state.followup_questions]
                st.session_state.step = 'processing'
                st.rerun()
    
    # =========================================================================
    # STEP 3: PROCESSING
    # =========================================================================
    elif st.session_state.step == 'processing':
        st.markdown("### ⚙️ Analyzing Your Research Idea")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress callback for real-time updates
        def update_progress(message: str, progress: float):
            if progress >= 0:
                progress_bar.progress(progress)
                status_text.markdown(f"**{message}**")
            else:
                status_text.error(message)
        
        # Set callback
        st.session_state.pipeline.progress_callback = update_progress
        
        try:
            # Step 0: Reality check (NEW)
            st.session_state.pipeline.run_reality_check(st.session_state.user_idea)
            
            # Process answers
            if st.session_state.followup_answers:
                st.session_state.pipeline.process_answers(st.session_state.followup_answers)
            
            # Run remaining pipeline
            st.session_state.pipeline.search_papers()
            st.session_state.pipeline.process_papers()
            st.session_state.pipeline.run_layer1_analysis()
            result = st.session_state.pipeline.run_layer2_analysis()
            
            st.session_state.result = result
            st.session_state.step = 'results'
            st.rerun()
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.exception("Pipeline error")
            if st.button("← Start Over"):
                st.session_state.step = 'input'
                st.rerun()
    
    # =========================================================================
    # STEP 4: RESULTS
    # =========================================================================
    elif st.session_state.step == 'results':
        result = st.session_state.result
        
        if result is None:
            st.error("No results available")
            if st.button("← Start Over"):
                st.session_state.step = 'input'
                st.rerun()
            return
        
        # Check if viewing matches for a sentence
        if st.session_state.selected_sentence_idx is not None:
            render_matches_panel(
                st.session_state.pipeline,
                st.session_state.selected_sentence_idx,
                result.sentence_annotations
            )
            return
        
        # NEW: Show reality check warning if exists
        if st.session_state.pipeline and st.session_state.pipeline.state.reality_check_warning:
            st.warning(st.session_state.pipeline.state.reality_check_warning)
            
            # Show existing examples if available
            rc = st.session_state.pipeline.state.reality_check_result
            if rc and rc.get('existing_examples'):
                with st.expander("📋 Similar Existing Products/Research", expanded=True):
                    for ex in rc['existing_examples'][:5]:
                        st.markdown(f"**{ex.get('name', 'Unknown')}** - Similarity: {ex.get('similarity', 0):.0%}")
                        st.markdown(f"*{ex.get('description', '')}*")
                        st.markdown("---")
        
        # Main results view
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge meter
            render_gauge(result.global_originality_score)
            
            # Summary
            st.markdown("### 📊 Summary")
            st.markdown(result.summary)
            
            # Stats
            st.markdown("### 📈 Statistics")
            stats = st.session_state.pipeline.get_stats()
            st.markdown(f"- **Papers analyzed:** {stats['papers_processed']}")
            st.markdown(f"- **Chunks indexed:** {stats['total_chunks']}")
            st.markdown(f"- **Keywords used:** {len(stats['keywords'])}")
            
            # Criteria breakdown
            if result.aggregated_criteria:
                st.markdown("### 🎯 Criteria Breakdown")
                criteria = result.aggregated_criteria
                st.progress(1 - criteria.problem_similarity, text=f"Problem: {(1-criteria.problem_similarity)*100:.0f}%")
                st.progress(1 - criteria.method_similarity, text=f"Method: {(1-criteria.method_similarity)*100:.0f}%")
                st.progress(1 - criteria.domain_overlap, text=f"Domain: {(1-criteria.domain_overlap)*100:.0f}%")
                st.progress(1 - criteria.contribution_similarity, text=f"Contribution: {(1-criteria.contribution_similarity)*100:.0f}%")
        
        with col2:
            st.markdown("### 📝 Your Idea Analysis")
            st.markdown("*Click 🔍 on highlighted sentences to see matching sources*")
            
            render_sentence_with_highlighting(result.sentence_annotations)
        
        # Cost breakdown
        st.markdown("---")
        st.markdown("### 💰 Cost Breakdown")
        render_cost_breakdown(result.cost)
        
        # Actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 New Analysis"):
                # Reset state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            # Download report
            report_text = f"""# Originality Assessment Report

## Score: {result.global_originality_score}/100

## Summary
{result.summary}

## Sentence Analysis
"""
            for ann in result.sentence_annotations:
                label_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(ann.label.value, "⚪")
                report_text += f"\n{label_emoji} [{ann.overlap_score:.0%} overlap] {ann.sentence}\n"
            
            st.download_button(
                "📥 Download Report",
                report_text,
                file_name="originality_report.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()

