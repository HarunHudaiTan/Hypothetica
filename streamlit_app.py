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
    </style>
    """, unsafe_allow_html=True)

    st.title("Hypothetica Research Assistant")
    st.markdown("""
    Welcome to the **Hypothetica Research Assistant**. 
    Enter your research idea below, and the system will:
    1. Search for relevant papers on arXiv
    2. Analyze their content using AI agents
    3. Generate a comprehensive research report
    """)

    # Initialize session state
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    with st.form("research_form"):
        user_idea = st.text_area("Enter your research idea:", height=150, 
                                placeholder="e.g., Graph Neural Networks for Drug Discovery: Novel architectures for molecular property prediction...")
        submitted = st.form_submit_button("Generate Analysis Report", disabled=st.session_state.processing)

    if submitted:
        if not user_idea:
            st.error("Please enter a research idea.")
        else:
            st.session_state.processing = True
            try:
                with st.status("Running Research Pipeline...", expanded=True) as status:
                    
                    # Step 1: Get papers using ArxivReq
                    status.write("Step 1: Fetching papers from ArXiv...")
                    arxiv_req = ArxivReq()
                    papers_json = arxiv_req.get_papers(user_idea)
                    papers = json.loads(papers_json)
                    
                    # Count total papers across all topics
                    num_papers = 0
                    if isinstance(papers, dict):
                        for topic_data in papers.values():
                            if isinstance(topic_data, dict) and 'papers' in topic_data:
                                num_papers += len(topic_data['papers'])
                    
                    status.write(f"Found {num_papers} papers.")
                    
                    # Step 2: Embed query and search literature
                    status.write("Step 2: Searching literature using embeddings...")
                    query_wrapper = QueryWrapper()
                    search_results = query_wrapper.search_literature(user_idea, include_scores=False)
                    status.write("Literature search completed.")
                    
                    # Step 3: Select relevant papers using agent
                    status.write("Step 3: Selecting most relevant papers...")
                    paper_selector = RelevantPaperSelectorAgent()
                    relevant_papers_json = paper_selector.generate_relevant_paper_selector_response(user_idea, search_results)
                    relevant_papers = json.loads(relevant_papers_json)
                    status.write(f"Selected {len(relevant_papers)} relevant papers.")

                    # Get PDF URLs and paper data
                    pdf_urls = []
                    paper_data = []
                    for paper in relevant_papers:
                        if 'url' in paper:
                            pdf_url = paper['url'].replace('/abs/', '/pdf/')
                            pdf_urls.append(pdf_url)
                            paper_data.append({
                                'title': paper.get('title', ''),
                                'abstract': paper.get('abstract', ''),
                                'url': pdf_url
                            })

                    # Process PDFs
                    status.write("Step 4: Processing PDFs (Downloading & Extracting)...")
                    heading_extractor = HeadingExtractor()
                    all_paper_data = []
                    
                    progress_bar = st.progress(0)
                    for idx, paper in enumerate(paper_data):
                        try:
                            status.write(f"Processing PDF: {paper['title']}...")
                            markdown = heading_extractor.convert_to_markdown(paper['url'])
                            headings = heading_extractor.extract_headings(markdown)
                            all_paper_data.append({'headings': headings, 'markdown': markdown})
                        except Exception as e:
                            status.write(f"Error processing {paper['url']}: {e}")
                            all_paper_data.append({'headings': [], 'markdown': ''})
                        progress_bar.progress((idx + 1) / len(paper_data))
                    
                    # Select headings
                    status.write("Step 5: Selecting relevant sections from papers...")
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
                    
                    # Extract text and save
                    status.write("Step 6: Extracting specific content...")
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

                    # Step 7: Generate Report
                    status.write("Step 7: Generating comprehensive research report...")
                    report_generator = ReportGenerator()
                    
                    if txt_file_paths:
                        research_report = report_generator.generate_report_generator_agent_response(txt_file_paths)
                        st.session_state.report = research_report
                        status.write("Report generated successfully!")
                    else:
                        st.session_state.report = "No valid papers were processed to generate a report."
                        status.write("Warning: No txt files were generated.")
                    
                    status.update(label="Research Pipeline Completed", state="complete", expanded=False)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write(traceback.format_exc())
            finally:
                st.session_state.processing = False

    if st.session_state.report:
        st.divider()
        st.subheader("Research Analysis Report")
        
        with st.container():
            st.markdown(st.session_state.report)
        
        # Option to download report
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                label="Download Report",
                data=st.session_state.report,
                file_name="research_report.md",
                mime="text/markdown"
            )
        with col2:
            if st.button("Start New Research"):
                st.session_state.report = None
                st.rerun()

if __name__ == "__main__":
    main()

