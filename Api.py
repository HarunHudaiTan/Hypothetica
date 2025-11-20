from flask import Flask, request, jsonify
import json
import traceback
import asyncio

# Import all the required classes
from ArxivReq import ArxivReq
from embeddemo.embed_query_wrapper import QueryWrapper
from Agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent
from heading_extraction.heading_extractor import HeadingExtractor
from Agents.heading_selector_agent import HeadingSelectorAgent
from Agents.report_generator_agent import ReportGenerator

app = Flask(__name__)

@app.route('/research_pipeline', methods=['POST'])
def research_pipeline():
    """
    Complete research pipeline endpoint that takes a user's research idea
    and generates a comprehensive research report.
    
    Expected input: JSON with 'user_idea' field
    Returns: JSON with the final research report
    """
    try:
        # Get user input
        data = request.get_json()
        if not data or 'user_idea' not in data:
            return jsonify({'error': 'Missing user_idea in request body'}), 400
        
        user_idea = data['user_idea']
        print(f"Processing research idea: {user_idea[:100]}...")
        
        # Step 1: Get papers using ArxivReq
        print("Step 1: Fetching papers from ArXiv...")
        arxiv_req = ArxivReq()
        papers_json = arxiv_req.get_papers(user_idea)
        papers = json.loads(papers_json)
        print(f"Found {len(papers.get('papers', []))} papers")
        print("Papers found:" + f"{papers}")
        
        # Step 2: Embed query and search literature
        print("Step 2: Searching literature using embeddings...")
        query_wrapper = QueryWrapper()
        search_results = query_wrapper.search_literature(user_idea, include_scores=False)
        print("Literature search completed")
        print("Search results:"+f"{search_results}")
        
        # Step 3: Select relevant papers using agent
        print("Step 3: Selecting most relevant papers...")
        paper_selector = RelevantPaperSelectorAgent()
        relevant_papers_json = paper_selector.generate_relevant_paper_selector_response(user_idea, search_results)
        relevant_papers = json.loads(relevant_papers_json)
        print( "relevant papers are :"f"{relevant_papers}")
        # Get PDF URLs and paper data from relevant papers
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

        # First: Process all PDFs synchronously one by one to get headings and markdowns
        heading_extractor = HeadingExtractor()
        all_paper_data = []
        
        for paper in paper_data:
            try:
                print(f"Processing PDF: {paper['url']}")
                markdown = heading_extractor.convert_to_markdown(paper['url'])
                headings = heading_extractor.extract_headings(markdown)
                all_paper_data.append({'headings': headings, 'markdown': markdown})
            except Exception as e:
                print(f"Error processing {paper['url']}: {e}")
                all_paper_data.append({'headings': [], 'markdown': ''})
        print("all paper data is :"+f"{all_paper_data}")
        # Second: Process each paper individually with heading selector
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
                print("selected headings:"+f"{selected_headings}")
                selected_headings_results.append(selected_headings)
        
        # Third: Extract text between headings for each paper and save to txt files
        final_results = []
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
                    print("extracted texts:"+f"{text}")
                    extracted_texts.append(text)
                
                # Save PDF information to text file
                saved_filepath = heading_extractor.save_pdf_info_to_txt(
                    paper_info=paper,
                    extracted_texts=extracted_texts,
                    user_idea=user_idea,
                    paper_index=i + 1
                )
                
                # Collect txt file paths for report generation
                if saved_filepath:
                    txt_file_paths.append(saved_filepath)
                
                final_results.append({
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'url': paper['url'],
                    'selected_headings': selected_headings,
                    'extracted_texts': extracted_texts,
                    'saved_to_file': saved_filepath
                })
        
        # Step 4: Generate comprehensive report using all txt files
        print("Step 4: Generating comprehensive research report...")
        report_generator = ReportGenerator()
        
        if txt_file_paths:
            research_report = report_generator.generate_report_generator_agent_response(txt_file_paths)
            print("Research report generated successfully")
        else:
            research_report = "No valid papers were processed to generate a report."
            print("Warning: No txt files were generated for report creation")
        
        return jsonify({
            'research_report': research_report
        })
        
    except Exception as e:
        print(f"Error in research pipeline: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Pipeline failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Research pipeline API is running'})

if __name__ == '__main__':
    print("Starting Research Pipeline API...")
    print("Available endpoints:")
    print("  POST /research_pipeline - Main research workflow")
    print("  GET  /health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5200)
