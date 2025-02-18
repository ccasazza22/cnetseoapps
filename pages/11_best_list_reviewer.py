from matplotlib import pyplot as plt
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import json
import tiktoken
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

class BestListAnalysis(BaseModel):
    user_perspective: bool
    writer_expertise: bool
    quantitative_measurements: bool
    competitor_comparisons: bool
    use_case_considerations: bool
    original_research: bool
    product_evolution: bool
    decision_factors: bool
    design_analysis: bool
    resource_links: bool
    best_justification: bool
    standalone_content: bool
    summary: str
    score: float  # Overall score based on criteria met
    recommendations: List[str]


api_key = st.secrets["dataforseoapikey"]["api_key"]


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets['openai']['openai_api_key'])

@st.cache_resource
def get_google_client():    
    return genai.Client(api_key=st.secrets["google"]["google_api"])

client = get_google_client()

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately chunk_size tokens"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

@st.cache_data
def get_embedding(text, model="text-embedding-004"):
    text = text.replace("\n", " ")
    result = client.models.embed_content(
    model=model,
    contents=text)
    return result.embeddings[0].values

def check_relevancy(chunk_embedding, query_embedding, threshold=0.85):
    """Check if chunk is relevant to search query"""
    similarity = cosine_similarity([chunk_embedding], [query_embedding])[0][0]
    return similarity > threshold

def check_best_list_best_practices(your_chunk, use_google=False):
    """Use LLM to verify true information gain"""
    
    prompt = f"""
    You are a content analyst determining the if the following reviews round up a.k.a "best list" follows best practices as outlined by Google below: 

    You are being given a chunk of the content you are analyzing: 
    {your_chunk}


    
    Evaluate the chunk based on the following criteria: 

    1. Does the chunk evaluate from a user's perspective.
    2. Does the chunk demonstrate that the writer is knowledgeable about what is being reviewed.
    3. Does the chunk share quantitative measurements about how something measures up in various categories of performance.
    4. Does the chunk explain what sets something apart from its competitors.
    5. Does the chunk cover comparable things to consider, or explain which might be best for certain uses or circumstances.
    6. Does the chunk discuss the benefits and drawbacks of something, based on the writer's own original research.
    7. Does the chunk describe how a product has evolved from previous models or releases to provide improvements, address issues, or otherwise help users in making a purchase decision.
    8. Does the chunk focus on the most important decision-making factors, based on the writer's experience or expertise (for example, a car review might determine that fuel economy and safety are key decision-making factors and rate performance in those areas).
    9. Does the chunk describe key choices in how a product has been designed and their effect on the users beyond what the manufacturer says.
    10. Does the chunk include links to other useful resources (your own or from other sites) to help a reader make a decision.
    11. Does the chunk  something as the best overall or the best for a certain purpose, include why you consider it the best, with first-hand supporting evidence.
    12. Does the chunk ensure there is enough useful content in your ranked lists for them to stand on their own, even if you choose to write separate in-depth single reviews.

    
 
    """

    if use_google:
        try:
            print("Starting Gemini analysis...")
            client = get_google_client()
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema':  BestListAnalysis,
                }
            )
            
            # Get parsed response
            analysis: BestListAnalysis = response.parsed
            return analysis.model_dump()
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            return None
    else:
        try:
            print("Starting OpenAI analysis...")
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            print(f"Error in OpenAI analysis: {str(e)}")
            return None

def get_page_content(url, timeout=30):
    """Fetch and extract main content and title from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = ' '.join(p.get_text().strip() for p in main_content.find_all('p') if p.get_text().strip())
            
            if not text:
                st.warning(f"No paragraph content found for {url}")
                # Try getting all text as fallback
                text = ' '.join(main_content.stripped_strings)
            
            print(f"Retrieved {len(text)} characters from {url}")
            
            if len(text) < 100:
                st.warning(f"Very short content ({len(text)} chars) from {url}")
            
            return {'title': title, 'content': text}
        st.error(f"No main content found for {url}")
        return None
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

def calculate_chunk_gain_score(analysis_result):
    """
    Calculate a score for a single chunk based on the LLM analysis results.
    Each 'True' criterion counts as 1 point.
    """
    if not analysis_result:
        return 0
    
    criteria = [
        'user_perspective',
        'writer_expertise',
        'quantitative_measurements',
        'competitor_comparisons',
        'use_case_considerations',
        'original_research',
        'product_evolution',
        'decision_factors',
        'design_analysis',
        'resource_links',
        'best_justification',
        'standalone_content'
    ]
    
    # Sum up the True values
    score = sum(1 for criterion in criteria if analysis_result.get(criterion, False))
    return score

def calculate_page_score(chunk_details):
    """
    Calculate the overall page score based on all chunk scores.
    Normalizes the final score to be between 1 and 10.
    """
    if not chunk_details:
        return 0
    
    # Get the maximum possible score per chunk (12 criteria)
    max_score_per_chunk = 12
    
    # Calculate the average score across all chunks
    total_score = sum(chunk['gain_score'] for chunk in chunk_details)
    num_chunks = len(chunk_details)
    average_score = total_score / (num_chunks * max_score_per_chunk)
    
    # Normalize to 1-10 scale
    normalized_score = 1 + (average_score * 9)  # This ensures a score between 1 and 10
    
    return round(normalized_score, 2)

def analyze_content(page_content, use_google=False):   
    chunks = chunk_text(page_content['content'])
    chunk_details = []
    
    print(f"Analyzing content: {page_content['title'][:50]}...")
    
    for chunk in chunks:
        # Only run LLM analysis once per chunk against all content
        llm_analysis = check_best_list_best_practices(chunk, use_google=use_google)
        chunk_gain = calculate_chunk_gain_score(llm_analysis)
        
        chunk_details.append({
            'text': chunk,
            'gain_score': chunk_gain,
            'analysis': llm_analysis
        })
    
    # Calculate the overall page score
    page_score = calculate_page_score(chunk_details)
    
    return {
        'page_score': page_score,
        'chunk_details': chunk_details,
        'title': page_content['title']
    }

import streamlit as st
from typing import List
import pandas as pd

def get_improvement_recommendations(analysis_result, use_google=False) -> str:
    """
    Use LLM to generate improvement recommendations based on failed criteria
    """
    # Extract failed criteria from all chunks
    failed_criteria = set()
    for chunk in analysis_result['chunk_details']:
        analysis = chunk['analysis']
        for criterion in [
            'user_perspective',
            'writer_expertise',
            'quantitative_measurements',
            'competitor_comparisons',
            'use_case_considerations',
            'original_research',
            'product_evolution',
            'decision_factors',
            'design_analysis',
            'resource_links',
            'best_justification',
            'standalone_content'
        ]:
            if not analysis.get(criterion, False):
                failed_criteria.add(criterion)
    
    if not failed_criteria:
        return "Great job! Your content meets all best practices criteria."
    
    prompt = f"""
    Analyze this best list content and provide specific recommendations for improvement.
    The content is missing or needs improvement in the following areas:
    {', '.join(failed_criteria)}
    
    Please provide detailed, actionable recommendations for how to improve the content
    in each of these areas. Format your response as a clear action plan.
    """
    
    if use_google:
        client = get_google_client()
        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt)
        return response.text
    else:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

def create_score_visualization(analysis_result):
    """
    Create a DataFrame for visualizing criteria scores across chunks
    """
    criteria_scores = []
    for chunk in analysis_result['chunk_details']:
        analysis = chunk['analysis']
        criteria_scores.append({
            'Chunk': len(criteria_scores) + 1,
            'User Perspective': analysis.get('user_perspective', False),
            'Writer Expertise': analysis.get('writer_expertise', False),
            'Quantitative Measurements': analysis.get('quantitative_measurements', False),
            'Competitor Comparisons': analysis.get('competitor_comparisons', False),
            'Use Case Considerations': analysis.get('use_case_considerations', False),
            'Original Research': analysis.get('original_research', False),
            'Product Evolution': analysis.get('product_evolution', False),
            'Decision Factors': analysis.get('decision_factors', False),
            'Design Analysis': analysis.get('design_analysis', False),
            'Resource Links': analysis.get('resource_links', False),
            'Best Justification': analysis.get('best_justification', False),
            'Standalone Content': analysis.get('standalone_content', False)
        })
    return pd.DataFrame(criteria_scores)

def main():
   
    st.title("Best List Content Analyzer")
    st.write("Analyze your best list content against Google's best practices")
    
    # URL input
    url = st.text_input("Enter the URL of your best list content:")
    use_google = st.checkbox("Use Google's Gemini (instead of GPT-4)", value=False)
    
    if st.button("Analyze Content"):
        if url:
            with st.spinner("Analyzing content..."):
                # Get page content
                page_content = get_page_content(url)
                
                if page_content:
                    # Analyze content
                    analysis_result = analyze_content(page_content, use_google=use_google)
                    
                    # Display results in columns
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.header("Overall Score")
                        score_color = "green" if analysis_result['page_score'] >= 8 else "orange" if analysis_result['page_score'] >= 6 else "red"
                        st.markdown(f"<h1 style='color: {score_color}'>{analysis_result['page_score']}/10</h1>", unsafe_allow_html=True)
                        
                        # Score interpretation
                        if analysis_result['page_score'] >= 8:
                            st.success("Excellent! Your content meets most best practices.")
                        elif analysis_result['page_score'] >= 6:
                            st.warning("Good, but there's room for improvement.")
                        else:
                            st.error("Needs significant improvement to meet best practices.")
                    
                    with col2:
                        st.header("Detailed Analysis")
                        # Create and display heatmap of criteria scores
                        df = create_score_visualization(analysis_result)
                        st.dataframe(df.style.applymap(lambda x: 'background-color: #90EE90' if x is True else 'background-color: #FFB6C1' if x is False else ''))
                    
                    # Get and display improvement recommendations
                    st.header("Improvement Recommendations")
                    recommendations = get_improvement_recommendations(analysis_result, use_google)
                    st.write(recommendations)
                    
                    # Show detailed chunk analysis in expander
                    with st.expander("View Chunk-by-Chunk Analysis"):
                        for i, chunk in enumerate(analysis_result['chunk_details'], 1):
                            st.subheader(f"Chunk {i}")
                            st.write(f"Score: {chunk['gain_score']}/12")
                            st.text(chunk['text'][:200] + "...")
                            st.json(chunk['analysis'])
                else:
                    st.error("Could not fetch content from the provided URL.")
        else:
            st.warning("Please enter a URL to analyze.")

if __name__ == "__main__":
    main()