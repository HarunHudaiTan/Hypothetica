import streamlit as st
import json
import traceback
import os

# Import all the required classes
from ArxivReq import ArxivReq
from embeddemo.embed_query_wrapper import QueryWrapper
from Agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent
from heading_extraction.heading_extractor import HeadingExtractor
from Agents.heading_selector_agent import HeadingSelectorAgent
from Agents.report_generator_agent import ReportGenerator

st.set_page_config(page_title="Hypothetica Research Assistant", layout="wide")


def main():
    st.markdown("""
    <style>
    .report-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stStatus {
        border-radius: 10px;
    }
    .pipeline-stats {
        background-color: #e8f4ea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Hypothetica Research Assistant")
    st.markdown("""
    Welcome to the **Hypothetica Research Assistant**. 
    Enter your research idea below, and the system will:
    1. Generate query variants and search arXiv (high-recall retrieval)
    2. Embedding similarity → Top 100 candidates
    3. Cross-encoder rerank → Top 20
    4. LLM selection → Final papers for analysis
    5. Generate comprehensive research report
    """)

    # Initialize session state
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'pipeline_stats' not in st.session_state:
        st.session_state.pipeline_stats = None

    # Advanced settings in sidebar
    with st.sidebar:
        st.header("⚙️ Pipeline Settings")
        
        papers_per_query = st.slider(
            "Papers per query variant",
            min_value=50, max_value=300, value=150, step=50,
            help="Number of papers to fetch from arXiv per query variant"
        )
        
        embedding_topk = st.slider(
            "Embedding search top-k",
            min_value=50, max_value=200, value=100, step=25,
            help="Number of candidates from embedding similarity search"
        )
        
        rerank_topk = st.slider(
            "Rerank top-k",
            min_value=20, max_value=50, value=30, step=5,
            help="Number of papers after cross-encoder reranking"
        )
        
        use_reranker = st.checkbox("Use cross-encoder reranking", value=True,
                                   help="Use cross-encoder for more accurate reranking")
        
        final_papers = st.slider(
            "Final papers to analyze",
            min_value=3, max_value=10, value=5,
            help="Number of papers for detailed analysis"
        )

    with st.form("research_form"):
        user_idea = st.text_area(
            "Enter your research idea:", 
            height=150,
            placeholder="e.g., Graph Neural Networks for Drug Discovery: Novel architectures for molecular property prediction..."
        )
        submitted = st.form_submit_button("Generate Analysis Report", disabled=st.session_state.processing)

    if submitted:
        if not user_idea:
            st.error("Please enter a research idea.")
        else:
            st.session_state.processing = True
            pipeline_stats = {}
            
            try:
                with st.status("Running Enhanced Research Pipeline...", expanded=True) as status:
                    
                    # ========================================
                    # Step 1: Generate Query Variants & Search ArXiv
                    # ========================================
                    status.write("**Step 1/7:** Generating query variants and searching arXiv...")
                    arxiv_req = ArxivReq()
                    
                    # This now uses QueryVariantAgent internally
                    papers_json = arxiv_req.get_papers(user_idea, papers_per_query=papers_per_query)
                    papers_summary = json.loads(papers_json)
                    
                    pipeline_stats['query_variants'] = len(papers_summary.get('query_variants', []))
                    pipeline_stats['total_fetched'] = papers_summary.get('total_papers_fetched', 0)
                    pipeline_stats['unique_papers'] = papers_summary.get('unique_papers', 0)
                    
                    status.write(f"✓ Generated {pipeline_stats['query_variants']} query variants")
                    status.write(f"✓ Fetched {pipeline_stats['total_fetched']} papers, {pipeline_stats['unique_papers']} unique after deduplication")
                    
                    # ========================================
                    # Step 2: Embedding Search + Reranking
                    # ========================================
                    status.write("**Step 2/7:** Running semantic search with embeddings...")
                    
                    query_wrapper = QueryWrapper(use_reranker=use_reranker)
                    search_results = query_wrapper.search_literature(
                        user_idea,
                        include_scores=True,
                        embedding_topk=embedding_topk,
                        rerank_topk=rerank_topk
                    )
                    
                    search_results_list = json.loads(search_results)
                    pipeline_stats['embedding_candidates'] = len(search_results_list) if isinstance(search_results_list, list) else 0
                    
                    status.write(f"✓ Semantic search returned {pipeline_stats['embedding_candidates']} candidates")
                    
                    # ========================================
                    # Step 3: LLM Final Selection (cross-encoder already did ranking)
                    # ========================================
                    status.write("**Step 3/7:** LLM selecting final papers from cross-encoder results...")
                    
                    paper_selector = RelevantPaperSelectorAgent()
                    relevant_papers_json = paper_selector.generate_relevant_paper_selector_response(
                        user_idea, 
                        search_results,
                        final_count=final_papers
                    )
                    relevant_papers = json.loads(relevant_papers_json)
                    
                    pipeline_stats['final_papers'] = len(relevant_papers)
                    status.write(f"✓ Selected {len(relevant_papers)} most relevant papers")
                    
                    # Display selected papers
                    if relevant_papers:
                        with st.expander("📚 Selected Papers", expanded=False):
                            for i, paper in enumerate(relevant_papers):
                                st.markdown(f"**{i+1}. {paper.get('title', 'No title')}**")
                                st.caption(f"Reason: {paper.get('selection_reason', 'N/A')}")
                                st.caption(f"URL: {paper.get('url', 'N/A')}")
                    
                    # ========================================
                    # Step 4: Process PDFs
                    # ========================================
                    status.write("**Step 4/7:** Processing PDFs (downloading & extracting)...")
                    
                    # Get PDF URLs and paper data
                    pdf_urls = []
                    paper_data = []
                    for paper in relevant_papers:
                        url = paper.get('url', '')
                        if url:
                            pdf_url = url.replace('/abs/', '/pdf/')
                            pdf_urls.append(pdf_url)
                            paper_data.append({
                                'title': paper.get('title', ''),
                                'abstract': paper.get('abstract', ''),
                                'url': pdf_url
                            })

                    heading_extractor = HeadingExtractor()
                    all_paper_data = []
                    
                    progress_bar = st.progress(0)
                    for idx, paper in enumerate(paper_data):
                        try:
                            status.write(f"  Processing: {paper['title'][:60]}...")
                            markdown = heading_extractor.convert_to_markdown(paper['url'])
                            headings = heading_extractor.extract_headings(markdown)
                            all_paper_data.append({'headings': headings, 'markdown': markdown})
                        except Exception as e:
                            status.write(f"  ⚠️ Error processing {paper['title'][:40]}: {e}")
                            all_paper_data.append({'headings': [], 'markdown': ''})
                        progress_bar.progress((idx + 1) / len(paper_data))
                    
                    pipeline_stats['pdfs_processed'] = sum(1 for d in all_paper_data if d['markdown'])
                    status.write(f"✓ Processed {pipeline_stats['pdfs_processed']} PDFs successfully")
                    
                    # ========================================
                    # Step 5: Select Relevant Sections
                    # ========================================
                    status.write("**Step 5/7:** Selecting relevant sections from papers...")
                    
                    heading_selector = HeadingSelectorAgent()
                    selected_headings_results = []
                    
                    for i, paper in enumerate(paper_data):
                        if i < len(all_paper_data):
                            paper_headings = all_paper_data[i]['headings']
                            headings_json = heading_extractor.get_headings_json(paper_headings)
                            title_and_abstract = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
                            
                            selected_headings = heading_selector.generate_heading_selector_agent_response(
                                user_idea, headings_json, title_and_abstract
                            )
                            selected_headings_results.append(selected_headings)
                    
                    status.write(f"✓ Section selection complete")
                    
                    # ========================================
                    # Step 6: Extract Content
                    # ========================================
                    status.write("**Step 6/7:** Extracting specific content...")
                    
                    txt_file_paths = []
                    
                    for i, paper in enumerate(paper_data):
                        if i < len(selected_headings_results) and i < len(all_paper_data):
                            markdown = all_paper_data[i]['markdown']
                            selected_headings = selected_headings_results[i]
                            extracted_texts = []
                            
                            for heading_interval in selected_headings:
                                text = heading_extractor.get_text_between_headings(
                                    markdown,
                                    heading_interval['from_heading'],
                                    heading_interval['to_heading']
                                )
                                extracted_texts.append(text)
                            
                            # Save PDF information to text file
                            saved_filepath = heading_extractor.save_pdf_info_to_txt(
                                paper_info=paper,
                                extracted_texts=extracted_texts,
                                user_idea=user_idea,
                                paper_index=i + 1
                            )
                            
                            if saved_filepath:
                                txt_file_paths.append(saved_filepath)
                    
                    pipeline_stats['txt_files'] = len(txt_file_paths)
                    status.write(f"✓ Extracted content to {len(txt_file_paths)} files")

                    # ========================================
                    # Step 7: Generate Report
                    # ========================================
                    status.write("**Step 7/7:** Generating comprehensive research report...")
                    
                    report_generator = ReportGenerator()
                    
                    if txt_file_paths:
                        research_report = report_generator.generate_report_generator_agent_response(txt_file_paths)
                        st.session_state.report = research_report
                        status.write("✓ Report generated successfully!")
                    else:
                        st.session_state.report = "No valid papers were processed to generate a report."
                        status.write("⚠️ Warning: No txt files were generated.")
                    
                    st.session_state.pipeline_stats = pipeline_stats
                    status.update(label="✅ Research Pipeline Completed", state="complete", expanded=False)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write(traceback.format_exc())
            finally:
                st.session_state.processing = False

    # Display pipeline stats
    if st.session_state.pipeline_stats:
        stats = st.session_state.pipeline_stats
        st.markdown("### 📊 Pipeline Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Query Variants", stats.get('query_variants', 0))
        with col2:
            st.metric("Papers Fetched", stats.get('unique_papers', 0))
        with col3:
            st.metric("After Reranking", stats.get('embedding_candidates', 0))
        with col4:
            st.metric("Final Selected", stats.get('final_papers', 0))

    # Display report
    if st.session_state.report:
        st.divider()
        st.subheader("📄 Research Analysis Report")
        
        with st.container():
            st.markdown(st.session_state.report)
        
        # Option to download report
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.report,
                file_name="research_report.md",
                mime="text/markdown"
            )
        with col2:
            if st.button("🔄 Start New Research"):
                st.session_state.report = None
                st.session_state.pipeline_stats = None
                st.rerun()


if __name__ == "__main__":
    main()
