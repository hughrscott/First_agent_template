# LLM-First Intelligent Nodes - Let AI do the thinking!

import os, re, base64, tempfile, json, math
import pandas as pd
import numpy as np
from typing import TypedDict, List, Dict, Any
from openai import OpenAI
from io import BytesIO, StringIO
import wikipedia
import chardet
import whisper
import requests

from agent.utils import download_file, get_youtube_transcript, extract_final_answer, get_file_type
from agent.config import SYSTEM_PROMPT, ATTACHMENTS
from duckduckgo_search import DDGS

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced State Definition
class AgentState(TypedDict):
    question: str
    answer: str
    extracted_data: str
    media_type: str
    attachment_id: str
    task_id: str
    question_analysis: dict
    tools_used: list

# SMART ROUTER (keep the one we have - it's working!)
def SmartRouter(state: AgentState) -> str:
    """Let the LLM decide what approach to take"""
    
    question = state["question"]
    attachment_info = ""
    
    if state.get("attachment_id") and state["attachment_id"] in ATTACHMENTS:
        attachment = ATTACHMENTS[state["attachment_id"]]
        attachment_info = f"Available attachment: {attachment['name']} (type: {attachment['type']})"
    else:
        attachment_info = "No attachment available"
    
    routing_prompt = f"""You are a task router. Analyze this question and choose the best approach.

Question: {question}
{attachment_info}

Available approaches:
1. web_search - for factual questions, research, current events
2. calculator - for mathematical calculations, number problems
3. data_analysis - for questions about CSV/Excel files or data processing
4. image_analysis - for questions about images or visual content
5. audio_analysis - for questions about audio files or transcripts
6. video_analysis - for questions about videos or YouTube content
7. multi_step - for complex questions needing multiple approaches

Choose exactly ONE approach that would be most effective for answering this question.
Respond with just the approach name (e.g., "web_search" or "calculator").
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        
        choice = response.choices[0].message.content.strip().lower()
        print(f"DEBUG: LLM chose '{choice}' for question: {question[:50]}...")
        
        route_map = {
            "web_search": "WebSearchNode",
            "calculator": "CalculatorNode", 
            "data_analysis": "DataExtractionNode",
            "image_analysis": "ImageExtractionNode",
            "audio_analysis": "AudioExtractionNode",
            "video_analysis": "VideoExtractionNode",
            "multi_step": "MultiStepNode"
        }
        
        return route_map.get(choice, "WebSearchNode")
        
    except Exception as e:
        print(f"Router error: {e}")
        if state.get("attachment_id") and state["attachment_id"] in ATTACHMENTS:
            attachment_type = ATTACHMENTS[state["attachment_id"]]["type"]
            fallback_map = {
                "audio": "AudioExtractionNode",
                "data": "DataExtractionNode",
                "image": "ImageExtractionNode", 
                "video": "VideoExtractionNode",
            }
            return fallback_map.get(attachment_type, "WebSearchNode")
        
        return "WebSearchNode"

# ENHANCED WEB SEARCH: Get full page content instead of snippets
def WebSearchNode(state: AgentState) -> AgentState:
    """Enhanced web search with full page content fetching"""
    try:
        question = state["question"]
        
        # Step 1: Let LLM plan search strategy
        search_planning_prompt = f"""You are a research expert. Plan how to search for this question:

Question: {question}

Create a search strategy:
1. Generate 2-3 different search queries that might find the answer
2. Consider what type of sources would be most reliable

Respond in JSON format:
{{"queries": ["query1", "query2", "query3"], "target_info": "what specific information to look for"}}"""

        planning_response = client.chat.completions.create(
            model="gpt-4o",  # Upgraded model
            messages=[{"role": "user", "content": search_planning_prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        
        try:
            search_plan = json.loads(planning_response.choices[0].message.content)
            queries = search_plan.get("queries", [question])
            target_info = search_plan.get("target_info", "")
        except:
            queries = [question]
            target_info = ""
        
        # Step 2: Search and collect URLs
        found_urls = []
        search_preview = ""
        
        for query in queries[:2]:  # Limit to 2 queries for speed
            try:
                with DDGS() as ddgs:
                    search_count = 0
                    for r in ddgs.text(query, region='wt-wt', safesearch='off', timelimit='year'):
                        found_urls.append(r['href'])
                        search_preview += f"Query: {query}\nTitle: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n\n"
                        search_count += 1
                        if search_count >= 3:  # Top 3 URLs per query
                            break
                if len(found_urls) >= 5:  # Total limit
                    break
            except Exception as e:
                print(f"Search error for query '{query}': {e}")
        
        # Step 3: Fetch full content from promising URLs
        full_content = ""
        successful_fetches = 0
        
        for url in found_urls[:3]:  # Fetch from top 3 URLs
            try:
                print(f"DEBUG: Attempting to fetch {url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                print(f"DEBUG: Response status: {response.status_code}")
                response.raise_for_status()
                
                # Extract text content (remove HTML tags)
                text_content = response.text
                print(f"DEBUG: Content length: {len(text_content)} characters")
                
                # Simple HTML tag removal
                import re
                text_content = re.sub(r'<script[^>]*>.*?</script>', '', text_content, flags=re.DOTALL)
                text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL)
                text_content = re.sub(r'<[^>]+>', ' ', text_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                print(f"DEBUG: Text content length after cleaning: {len(text_content)} characters")
                
                # Focus on relevant sections if target_info is available
                if target_info and len(text_content) > 4000:
                    # Look for paragraphs containing target keywords
                    paragraphs = text_content.split('\n')
                    relevant_paragraphs = []
                    target_words = target_info.lower().split()
                    
                    for para in paragraphs:
                        if any(word in para.lower() for word in target_words):
                            relevant_paragraphs.append(para)
                    
                    if relevant_paragraphs:
                        text_content = '\n'.join(relevant_paragraphs[:20])  # Top 20 relevant paragraphs
                
                # Limit content size
                if len(text_content) > 4000:
                    text_content = text_content[:4000] + "..."
                
                full_content += f"\n--- Content from {url} ---\n{text_content}\n"
                successful_fetches += 1
                print(f"DEBUG: ✅ Successfully processed {url}")
                
                if len(full_content) > 10000:  # Total content limit
                    break
                    
            except Exception as e:
                print(f"DEBUG: ❌ Failed to fetch {url}: {e}")
                continue
        
        # Step 4: Fallback to search snippets if no full content
        if not full_content.strip() or successful_fetches == 0:
            print("DEBUG: No full content fetched, using search snippets")
            full_content = search_preview
        else:
            print(f"DEBUG: Successfully fetched content from {successful_fetches} URLs")
        
        if not full_content.strip():
            state["answer"] = "Could not find relevant search results."
            return state

        # Step 5: Let LLM analyze the full content
        analysis_prompt = f"""You are a research analyst. Analyze this content to answer the question.

Original Question: {question}
Target Information: {target_info}

Content from web sources:
{full_content}

Instructions:
1. Carefully read through all the content
2. Extract the specific information that answers the question
3. Be precise with numbers, names, dates, etc.
4. If you find the answer, provide it clearly
5. If information is unclear, indicate what you found

{SYSTEM_PROMPT.strip()}"""

        response = client.chat.completions.create(
            model="gpt-4o",  # Use most capable model
            messages=[
                {"role": "system", "content": "You are a research analyst who provides precise, well-researched answers."},
                {"role": "user", "content": analysis_prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        
        raw_answer = response.choices[0].message.content
        state["answer"] = extract_final_answer(raw_answer)
        state["extracted_data"] = full_content[:1000] + "..." if len(full_content) > 1000 else full_content
        
        print(f"DEBUG: Enhanced web search found {len(found_urls)} URLs, fetched {successful_fetches} successfully")
        
    except Exception as e:
        state["answer"] = f"Web search error: {str(e)}"
        print(f"DEBUG: Web search error: {e}")
    
    return state

# LLM-FIRST DATA ANALYSIS: Let AI understand and analyze data
def DataExtractionNode(state: AgentState) -> AgentState:
    """Intelligent data analysis - let LLM understand the data and question"""
    try:
        question = state["question"]
        content = None
        file_ext = ""

        # Get the data
        if state.get("attachment_id") and state["attachment_id"] in ATTACHMENTS:
            attachment = ATTACHMENTS[state["attachment_id"]]
            content = attachment["content"]
            file_ext = os.path.splitext(attachment["name"])[1][1:].lower()
            print(f"DEBUG: Processing {attachment['name']} ({file_ext})")
        elif "http" in question:
            url_match = re.search(r"https?://\S+\.(csv|xlsx?|json)", question, re.I)
            if url_match:
                content = download_file(url_match.group(0))
                file_ext = url_match.group(1).lower()

        if not content:
            state["answer"] = "No data file available to analyze"
            return state

        # Handle Python files with LLM analysis
        if file_ext == "py":
            code_content = content.decode("utf-8", errors="replace")
            
            code_analysis_prompt = f"""Analyze this Python code and answer the question:

Question: {question}

Python Code:
```python
{code_content}
```

Instructions:
1. Read through the code carefully
2. Trace the execution step by step
3. Calculate what the final output would be
4. If the code has multiple outputs, identify which one is "final"

{SYSTEM_PROMPT.strip()}"""

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a Python code analyst. Trace code execution carefully."},
                    {"role": "user", "content": code_analysis_prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            
            raw_answer = response.choices[0].message.content
            state["answer"] = extract_final_answer(raw_answer)
            return state

        # Load data files
        df = None
        if file_ext == "csv":
            detected = chardet.detect(content)
            encoding = detected["encoding"] or "utf-8"
            decoded_content = content.decode(encoding, errors="replace")
            df = pd.read_csv(StringIO(decoded_content))
        elif file_ext in ("xls", "xlsx"):
            df = pd.read_excel(BytesIO(content))
        elif file_ext == "json":
            decoded_content = content.decode("utf-8", errors="replace")
            df = pd.read_json(StringIO(decoded_content))
        else:
            state["answer"] = f"Unsupported file format: {file_ext}"
            return state

        print(f"DEBUG: Data loaded: {df.shape} rows x columns")
        print(f"DEBUG: Columns: {list(df.columns)}")

        # Step 1: Let LLM understand the data structure and question
        data_preview = df.head(10).to_string()
        data_summary = f"""
Data Shape: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}

Sample Data (first 10 rows):
{data_preview}

Numeric Summary:
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"}
"""

        analysis_planning_prompt = f"""You are a data analyst. Analyze this question and data to determine what analysis is needed.

Question: {question}

Data Summary:
{data_summary}

Instructions:
1. Understand what the question is asking for
2. Identify which columns are relevant
3. Determine what calculations or operations are needed
4. Plan the analysis step by step

Respond in JSON format:
{{
    "analysis_type": "sum/count/average/filter/group_by/calculation",
    "relevant_columns": ["col1", "col2"],
    "steps": ["step 1", "step 2", "step 3"],
    "expected_result_type": "number/text/list"
}}"""

        planning_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": analysis_planning_prompt}],
            max_tokens=300,
            temperature=0.1,
        )

        try:
            analysis_plan = json.loads(planning_response.choices[0].message.content)
            print(f"DEBUG: Analysis plan: {analysis_plan}")
        except:
            analysis_plan = {"analysis_type": "general", "relevant_columns": [], "steps": []}

        # Step 2: Execute the analysis based on LLM's plan
        result = None
        
        # Try to execute common analysis patterns
        analysis_type = analysis_plan.get("analysis_type", "").lower()
        relevant_cols = analysis_plan.get("relevant_columns", [])
        
        if "sum" in analysis_type or "total" in question.lower():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if relevant_cols:
                target_cols = [col for col in relevant_cols if col in df.columns and col in numeric_cols]
            else:
                target_cols = numeric_cols
            
            if len(target_cols) > 0:
                # For sales questions, try to filter out drinks if mentioned
                if "food" in question.lower() and "drink" in question.lower():
                    # Look for category columns
                    category_cols = [col for col in df.columns if 'category' in col.lower() or 'type' in col.lower()]
                    if category_cols:
                        mask = ~df[category_cols[0]].str.contains('drink|beverage', case=False, na=False)
                        result = df[mask][target_cols[0]].sum()
                    else:
                        result = df[target_cols[0]].sum()
                else:
                    result = df[target_cols[0]].sum()
        
        elif "count" in analysis_type or "how many" in question.lower():
            if "unique" in question.lower() and relevant_cols:
                result = df[relevant_cols[0]].nunique()
            else:
                result = len(df)
        
        elif "average" in analysis_type or "mean" in question.lower():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if relevant_cols:
                target_cols = [col for col in relevant_cols if col in numeric_cols]
            else:
                target_cols = numeric_cols
            if len(target_cols) > 0:
                result = df[target_cols[0]].mean()

        # Step 3: If we got a result, format it properly
        if result is not None:
            if isinstance(result, float):
                if "USD" in question or "$" in question:
                    state["answer"] = f"{result:.2f}"
                elif result.is_integer():
                    state["answer"] = str(int(result))
                else:
                    state["answer"] = f"{result:.2f}".rstrip('0').rstrip('.')
            else:
                state["answer"] = str(result)
        else:
            # Step 4: Fall back to LLM analysis of the data
            fallback_prompt = f"""You are a data analyst. Answer this question using the provided data.

Question: {question}

Data Summary:
{data_summary}

Instructions:
1. Look at the data structure and understand what each column represents
2. Perform the necessary calculations to answer the question
3. Be precise and show your reasoning
4. If you need to filter, aggregate, or calculate, explain what you're doing

{SYSTEM_PROMPT.strip()}"""

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Provide precise answers based on data analysis."},
                    {"role": "user", "content": fallback_prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            
            raw_answer = response.choices[0].message.content
            state["answer"] = extract_final_answer(raw_answer)

        state["extracted_data"] = f"Analyzed {file_ext} file with {df.shape[0]} rows and {df.shape[1]} columns"
        
    except Exception as e:
        state["answer"] = f"Data analysis error: {str(e)}"
        print(f"DEBUG: Data analysis error: {e}")
    
    return state

# LLM-FIRST CALCULATOR: Let AI understand math problems
def CalculatorNode(state: AgentState) -> AgentState:
    """Intelligent calculator - let LLM understand and solve math problems"""
    try:
        question = state["question"]
        
        math_prompt = f"""You are a mathematical expert. Solve this problem step by step.

Question: {question}

Instructions:
1. Identify what type of mathematical problem this is
2. Break down the problem into steps
3. Perform the calculations carefully
4. Double-check your work
5. Provide the final numerical answer

If this involves:
- Tables or matrices: analyze the structure and perform the required operations
- Word problems: extract the numbers and operations needed
- Algebraic problems: solve systematically
- Logic problems: work through the logic step by step

Show your work clearly and provide the final answer.

{SYSTEM_PROMPT.strip()}"""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a mathematical expert. Solve problems step by step with precision."},
                {"role": "user", "content": math_prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )
        
        raw_answer = response.choices[0].message.content
        state["answer"] = extract_final_answer(raw_answer)
        state["extracted_data"] = raw_answer
        
    except Exception as e:
        state["answer"] = f"Calculation error: {str(e)}"
    
    return state

# LLM-FIRST MULTI-STEP: Let AI plan and execute complex workflows
def MultiStepNode(state: AgentState) -> AgentState:
    """Intelligent multi-step processing - let LLM plan and orchestrate"""
    try:
        question = state["question"]
        
        # Step 1: Let LLM create a detailed plan
        planning_prompt = f"""You are a task planning expert. This question requires multiple steps to solve.

Question: {question}

Available tools:
- web_search: can search the internet for information
- data_analysis: can analyze CSV/Excel files  
- calculation: can perform mathematical operations
- reasoning: can analyze and synthesize information

Create a detailed step-by-step plan to answer this question:
1. What information do you need to find?
2. What tools should be used in what order?
3. How will you combine the results?

Respond in JSON format:
{{
    "steps": [
        {{"step": 1, "action": "web_search", "goal": "find specific information", "query": "search query"}},
        {{"step": 2, "action": "calculation", "goal": "perform calculation", "operation": "what to calculate"}},
        {{"step": 3, "action": "reasoning", "goal": "synthesize results", "method": "how to combine"}}
    ],
    "final_goal": "what the final answer should contain"
}}"""

        planning_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        
        try:
            plan = json.loads(planning_response.choices[0].message.content)
            steps = plan.get("steps", [])
        except:
            # Fallback: try web search then reasoning
            steps = [
                {"step": 1, "action": "web_search", "goal": "research the question"},
                {"step": 2, "action": "reasoning", "goal": "analyze and answer"}
            ]

        # Step 2: Execute the plan
        step_results = []
        for i, step in enumerate(steps[:4]):  # Limit to 4 steps
            action = step.get("action", "web_search")
            goal = step.get("goal", "")
            
            print(f"DEBUG: Executing step {i+1}: {action} - {goal}")
            
            if action == "web_search":
                # Execute web search step
                temp_state = state.copy()
                if "query" in step:
                    temp_state["question"] = step["query"]
                temp_state = WebSearchNode(temp_state)
                step_results.append(f"Step {i+1} ({action}): {temp_state['answer']}")
                
            elif action == "calculation":
                # Execute calculation step
                temp_state = state.copy()
                temp_state = CalculatorNode(temp_state)
                step_results.append(f"Step {i+1} ({action}): {temp_state['answer']}")
                
            elif action == "data_analysis":
                # Execute data analysis step
                temp_state = state.copy()
                temp_state = DataExtractionNode(temp_state)
                step_results.append(f"Step {i+1} ({action}): {temp_state['answer']}")
        
        # Step 3: Let LLM synthesize all results
        synthesis_prompt = f"""You are a synthesis expert. Combine these step results to answer the original question.

Original Question: {question}

Step Results:
{chr(10).join(step_results)}

Instructions:
1. Review all the step results
2. Identify which results are most relevant to the original question
3. Combine or calculate as needed to get the final answer
4. Ensure your answer directly addresses the original question

{SYSTEM_PROMPT.strip()}"""

        synthesis_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a synthesis expert. Provide precise final answers based on step results."},
                {"role": "user", "content": synthesis_prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        
        raw_answer = synthesis_response.choices[0].message.content
        state["answer"] = extract_final_answer(raw_answer)
        state["extracted_data"] = f"Multi-step execution: {chr(10).join(step_results)}"
        
    except Exception as e:
        state["answer"] = f"Multi-step error: {str(e)}"
    
    return state

# KEEP existing media nodes but make them LLM-first too
def ImageExtractionNode(state: AgentState) -> AgentState:
    """LLM-first image analysis"""
    try:
        content = None

        if state.get("attachment_id") and state["attachment_id"] in ATTACHMENTS:
            content = ATTACHMENTS[state["attachment_id"]]["content"]
        elif "http" in state["question"]:
            url_match = re.search(r"https?://\S+\.(jpg|jpeg|png|gif)", state["question"], re.I)
            if url_match:
                content = download_file(url_match.group(0))

        if not content:
            state["answer"] = "No image available to analyze"
            return state

        base64_image = base64.b64encode(content).decode()

        # Enhanced prompt for better image analysis
        enhanced_prompt = f"""Analyze this image carefully to answer the question.

Question: {state['question']}

Instructions:
1. Look at the image in detail
2. Identify all relevant elements that relate to the question
3. If this is a chess position, analyze the board state and possible moves
4. If this is a chart/graph, read the data carefully
5. Provide a precise answer based on what you can see

{SYSTEM_PROMPT.strip()}"""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert image analyst. Analyze images carefully and precisely."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            max_tokens=400,
            temperature=0.1,
        )
        raw_answer = response.choices[0].message.content
        state["answer"] = extract_final_answer(raw_answer)
        
    except Exception as e:
        state["answer"] = f"Image analysis error: {str(e)}"
    
    return state

def AudioExtractionNode(state: AgentState) -> AgentState:
    """LLM-first audio analysis"""
    try:
        content = None

        if state.get("attachment_id") and state["attachment_id"] in ATTACHMENTS:
            content = ATTACHMENTS[state["attachment_id"]]["content"]

        if not content:
            state["answer"] = "No audio file available to analyze"
            return state

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(content)
            tmp.flush()
            
            # Use whisper to transcribe
            model = whisper.load_model("base")
            result = model.transcribe(tmp.name)
            transcription = result["text"]

            # Enhanced prompt for better audio analysis
            enhanced_prompt = f"""Analyze this audio transcription to answer the question.

Question: {state['question']}

Audio Transcription:
{transcription}

Instructions:
1. Read through the transcription carefully
2. Extract the specific information requested in the question
3. If looking for ingredients, list only the ingredients mentioned
4. If looking for page numbers, extract only the numbers
5. Format your answer according to the question requirements

{SYSTEM_PROMPT.strip()}"""

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing audio transcriptions. Extract precise information."},
                    {"role": "user", "content": enhanced_prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            raw_answer = response.choices[0].message.content
            state["answer"] = extract_final_answer(raw_answer)
            
    except Exception as e:
        state["answer"] = f"Audio processing error: {str(e)}"
    
    return state

def VideoExtractionNode(state: AgentState) -> AgentState:
    """Enhanced video analysis - transcript OR audio extraction"""
    try:
        video_content = None
        video_url = None
        
        # Check for YouTube URL in question
        youtube_match = re.search(r"https?://www\.youtube\.com/watch\?v=[a-zA-Z0-9_-]+", state["question"])
        
        if youtube_match:
            video_url = youtube_match.group(0)
            print(f"DEBUG: Found YouTube URL: {video_url}")
            
            # Method 1: Try transcript first (fast)
            transcript = get_youtube_transcript(video_url)
            if transcript:
                print(f"DEBUG: Got transcript ({len(transcript)} chars)")
                
                enhanced_prompt = f"""Analyze this video transcript to answer the question.

Question: {state['question']}

Video Transcript:
{transcript}

Instructions:
1. Read through the entire transcript carefully
2. Look for the specific information requested
3. If looking for dialogue or quotes, find the exact words
4. If counting elements, go through systematically
5. Provide the precise answer requested

{SYSTEM_PROMPT.strip()}"""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing video content. Extract precise information from transcripts."},
                        {"role": "user", "content": enhanced_prompt},
                    ],
                    max_tokens=400,
                    temperature=0.1,
                )
                raw_answer = response.choices[0].message.content
                state["answer"] = extract_final_answer(raw_answer)
                return state
            
            # Method 2: If no transcript, try downloading and extracting audio
            print("DEBUG: No transcript available, attempting video download and audio extraction")
            
            try:
                # Download the video
                import yt_dlp
                
                ydl_opts = {
                    'format': 'best[height<=720]',  # Limit quality for faster download
                    'noplaylist': True,
                    'quiet': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(video_url, download=False)
                    video_title = info.get('title', 'Unknown')
                    print(f"DEBUG: Video title: {video_title}")
                    
                    # Download to temporary location
                    with tempfile.TemporaryDirectory() as temp_dir:
                        ydl_opts['outtmpl'] = f'{temp_dir}/video.%(ext)s'
                        
                        # Download the video
                        ydl.download([video_url])
                        
                        # Find the downloaded file
                        import glob
                        video_files = glob.glob(f'{temp_dir}/video.*')
                        if video_files:
                            video_file_path = video_files[0]
                            print(f"DEBUG: Downloaded video to {video_file_path}")
                            
                            # Read video content
                            with open(video_file_path, 'rb') as f:
                                video_content = f.read()
                            
                            print(f"DEBUG: Video file size: {len(video_content)} bytes")
            
            except Exception as e:
                print(f"DEBUG: Video download failed: {e}")
                # Try alternative download method or fallback
                state["answer"] = "Video download failed - unable to analyze"
                return state
        
        # If we have video content, extract audio and transcribe
        if video_content:
            print("DEBUG: Attempting audio extraction from video")
            
            try:
                # Extract audio from video
                from agent.utils import extract_audio_from_video
                audio_content = extract_audio_from_video(video_content)
                
                if audio_content:
                    print(f"DEBUG: Extracted audio ({len(audio_content)} bytes)")
                    
                    # Use Whisper to transcribe the audio
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                        tmp.write(audio_content)
                        tmp.flush()
                        
                        model = whisper.load_model("base")
                        result = model.transcribe(tmp.name)
                        transcription = result["text"]
                        
                        print(f"DEBUG: Whisper transcription ({len(transcription)} chars): {transcription[:100]}...")
                        
                        # Analyze the transcription
                        enhanced_prompt = f"""Analyze this video audio transcription to answer the question.

Question: {state['question']}

Audio Transcription:
{transcription}

Instructions:
1. Read through the transcription carefully
2. Extract the specific information requested in the question
3. If looking for dialogue, quotes, or specific words, find them precisely
4. If counting elements, go through systematically
5. Provide the exact answer requested

{SYSTEM_PROMPT.strip()}"""

                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are an expert at analyzing audio transcriptions from video content."},
                                {"role": "user", "content": enhanced_prompt},
                            ],
                            max_tokens=400,
                            temperature=0.1,
                        )
                        raw_answer = response.choices[0].message.content
                        state["answer"] = extract_final_answer(raw_answer)
                        
                else:
                    state["answer"] = "Audio extraction failed"
                    
            except Exception as e:
                print(f"DEBUG: Audio extraction/transcription error: {e}")
                state["answer"] = f"Audio processing error: {str(e)}"
        else:
            state["answer"] = "No video content available for analysis"
            
    except Exception as e:
        state["answer"] = f"Video processing error: {str(e)}"
        print(f"DEBUG: Video processing error: {e}")
    
    return state
# Keep the existing AnswerRefinementNode - it's already LLM-first
def AnswerRefinementNode(state: AgentState) -> AgentState:
    try:
        question = state["question"]
        initial_answer = state["answer"]
        extracted_data = state.get("extracted_data", "")

        refinement_prompt = f"""
        Original Question: {question}
        Initial Answer: {initial_answer}
        Extracted Context/Data: {extracted_data if extracted_data else "No specific data was extracted, the answer was generated based on general knowledge or initial processing."}

        Your task is to critically review the Initial Answer in the context of the Original Question and Extracted Context/Data.
        Refine the Initial Answer to ensure it is accurate, directly answers the question, and strictly follows the FINAL ANSWER formatting rules.
        If the Initial Answer seems correct and appropriately formatted, you can simply re-state it.
        If the Initial Answer is "unknown" or an error message, try to re-evaluate the question using the available context to provide a valid answer if possible.

        Strict FINAL ANSWER formatting rules:
        - A number OR
        - As few words as possible OR
        - A comma separated list of numbers and/or strings

        Specific formatting rules:
        1. For numbers:
           - Don't use commas (e.g., 1000000 not 1,000,000)
           - Don't include units ($, %, etc.) unless specified
        2. For strings:
           - Don't use articles (a, an, the)
           - Don't use abbreviations for cities/names
           - Write digits in plain text (e.g., "two" instead of "2")
        3. For comma-separated lists:
           - Apply the above rules to each element
           - Separate elements with commas only (no spaces unless part of the element)

        Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": refinement_prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        refined_raw_answer = response.choices[0].message.content
        state["answer"] = extract_final_answer(refined_raw_answer)
        print(f"  Refinement Node: Initial Answer - '{initial_answer}', Refined Answer - '{state['answer']}'")
    except Exception as e:
        state["answer"] = f"Refinement error: {str(e)}"
        print(f"  Refinement Node Error: {e}")
    return state