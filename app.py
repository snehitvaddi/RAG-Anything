import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import os
from dotenv import load_dotenv
import json
from langchain.chains import LLMChain
from typing import Literal, Dict, Any

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Food Safety Incidents Analysis",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    /* Fixed chat input styling */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem 2rem;
        z-index: 1000;
        border-top: 1px solid #ddd;
    }
    /* Add padding to main container to prevent content from being hidden behind input */
    .main .block-container {
        padding-bottom: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

class QueryRouter:
    def __init__(self, llm):
        self.llm = llm
        self.router_prompt = PromptTemplate(
            template="""You are an expert at analyzing questions about food safety incidents. 
            Determine the best way to answer the given question by choosing one of these options:
            1. SQL - for questions requiring statistical analysis, counts, aggregations, or structured data queries
            2. RAG - for questions about specific incidents, detailed descriptions, or narrative information
            3. NONE - for questions not related to food safety incidents

            Question: {question}

            Think step by step:
            1. Is this question about food safety incidents?
            2. Does it require numerical analysis or aggregation?
            3. Does it ask for specific details or descriptions?

            Return only one word (SQL/RAG/NONE) as your decision.

            Decision:""",
            input_variables=["question"]
        )
        self.chain = LLMChain(llm=llm, prompt=self.router_prompt)

    def route_query(self, question: str) -> str:
        """Route the query to appropriate chain"""
        response = self.chain.run(question).strip().upper()
        return response if response in ['SQL', 'RAG', 'NONE'] else 'NONE'

class IncidentQueryEngine:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.qa_prompt = PromptTemplate(
            template="""You are a food safety incident analysis expert. Based on the provided context, 
            please answer the question thoroughly and accurately. If you cannot find the answer in the context, say so.

            Context from food safety incidents database:
            {context}

            Question: {question}

            Please provide a detailed answer, including:
            1. Direct relevant information from the incidents
            2. Any patterns or trends you notice
            3. Specific examples when applicable

            Answer:""",
            input_variables=["context", "question"]
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

    def get_answer(self, question: str, num_chunks: int = 4):
        """Get an answer for a specific question"""
        # Get relevant documents from vector store
        relevant_docs = self.vector_store.similarity_search(question, k=num_chunks)
        
        # Combine the content from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Get the answer using the LLM
        response = self.qa_chain.run(context=context, question=question)
        
        return {
            "answer": response,
            "sources": [doc.metadata for doc in relevant_docs]
        }

class IncidentAnalyzer:
    def __init__(self):
        # Initialize OpenAI components
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4",
            temperature=0.2
        )
        
        # Initialize Router
        self.router = QueryRouter(self.llm)
        
        # Initialize SQL Chain
        self.sql_chain = self._initialize_sql_chain()
        
        # Initialize RAG Chain
        self.rag_chain = self._initialize_rag_chain()

    def _initialize_rag_chain(self):
        """Initialize RAG chain"""
        try:
            # Load the vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(
                "food_safety_store.faiss",
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Create and return the IncidentQueryEngine instance
            return IncidentQueryEngine(vector_store=vector_store, llm=self.llm)
        except Exception as e:
            st.error(f"Error initializing RAG chain: {str(e)}")
            return None

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query using appropriate chain"""
        try:
            # Route the query
            route = self.router.route_query(query)
            
            if route == "SQL":
                return self._process_sql_query(query)
            elif route == "RAG":
                if self.rag_chain is None:
                    return {
                        'success': False,
                        'type': 'rag',
                        'message': "RAG chain is not properly initialized"
                    }
                return self._process_rag_query(query)
            else:
                return {
                    'success': True,
                    'type': 'none',
                    'answer': "I apologize, but this question doesn't appear to be related to food safety incidents. Please ask a question about food safety recalls, contamination incidents, or related topics."
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Error processing query: {str(e)}"
            }

    def _process_rag_query(self, query: str) -> Dict[str, Any]:
        """Process RAG-type query"""
        try:
            result = self.rag_chain.get_answer(query)
            return {
                'success': True,
                'type': 'rag',
                'answer': result['answer'],
                'sources': result['sources']
            }
        except Exception as e:
            return {
                'success': False,
                'type': 'rag',
                'message': f"Error processing RAG query: {str(e)}"
            }

    def _process_sql_query(self, query: str) -> Dict[str, Any]:
        """Process SQL-type query"""
        result = process_query(self.sql_chain, query)  # Using existing process_query function
        if result['success']:
            return {
                'success': True,
                'type': 'sql',
                'sql_query': result.get('sql_query'),
                'answer': result.get('answer')
            }
        return result

    def _process_rag_query(self, query: str) -> Dict[str, Any]:
        """Process RAG-type query"""
        try:
            result = self.rag_chain.get_answer(query)
            return {
                'success': True,
                'type': 'rag',
                'answer': result['answer'],
                'sources': result['sources']
            }
        except Exception as e:
            return {
                'success': False,
                'type': 'rag',
                'message': f"Error processing RAG query: {str(e)}"
            }

    def _initialize_sql_chain(self):
        """Initialize SQL database chain"""
        try:
            db = SQLDatabase.from_uri(
                f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}",
                sample_rows_in_table_info=3
            )
            
            return SQLDatabaseChain.from_llm(
                llm=self.llm,
                db=db,
                verbose=True,
                return_direct=False,
                return_intermediate_steps=True
            )
        except Exception as e:
            st.error(f"Error initializing SQL chain: {str(e)}")
            return None

def process_query(chain, query):
    """Process a query using invoke instead of run"""
    try:
        # Use invoke instead of run
        response = chain.invoke({
            "query": query
        })
        
        # Extract components from the response
        if isinstance(response, dict):
            sql_query = None
            result = None
            
            # Get SQL query from intermediate steps
            if 'intermediate_steps' in response:
                for step in response['intermediate_steps']:
                    if isinstance(step, str) and 'SELECT' in step.upper():
                        sql_query = step
                        break
            
            # Get final result
            if 'result' in response:
                result = response['result']
            
            return {
                'success': True,
                'sql_query': sql_query,
                'answer': result
            }
        else:
            return {
                'success': False,
                'message': "Unexpected response format"
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Error processing query: {str(e)}"
        }

def format_answer(raw_answer):
    """Format the answer for better display"""
    try:
        if not raw_answer:
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        # Clean up the answer
        answer = raw_answer.strip()
        
        # Remove any SQL-related content if present
        if 'SQLQuery:' in answer:
            answer = answer.split('SQLQuery:')[0].strip()
        
        # Format list-like responses
        lines = answer.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if any(line.startswith(str(i) + '.') for i in range(10)):
                    formatted_lines.append(f"* {line}")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except Exception:
        return raw_answer if raw_answer else "Error formatting response"

def format_rag_answer(result: dict):
    """Format RAG answer with sources"""
    formatted_text = [result['answer']]
    
    if result.get('sources'):
        formatted_text.append("\n\n**Sources Used:**")
        for i, source in enumerate(result['sources'], 1):
            formatted_text.append(f"\n**Source {i}:**")
            for key, value in source.items():
                formatted_text.append(f"- {key}: {value}")
    
    return "\n".join(formatted_text)

def main():
    st.title("RAG Anything: PDF, Docx, SQL or whatever")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = IncidentAnalyzer()
    
    if st.session_state.analyzer is None:
        st.error("Failed to initialize the application. Please check the logs.")
        return

    # Add evaluation button in the top right
    with st.container():
        col1, col2 = st.columns([8, 2])
        with col2:
            show_eval = st.button("📊 Show Evaluation")

    # Display evaluation metrics if button is clicked
    if show_eval:
        with st.expander("Evaluation Metrics", expanded=True):
            try:
                with open('evaluation_results.json', 'r') as f:
                    eval_results = json.load(f)
                
                # Display metrics
                st.markdown("### Overall Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Success Rate", f"{eval_results['metrics']['success_rate']*100:.1f}%")
                with col2:
                    st.metric("ROUGE-1", f"{eval_results['metrics']['avg_rouge1']:.3f}")
                with col3:
                    st.metric("ROUGE-2", f"{eval_results['metrics']['avg_rouge2']:.3f}")
                with col4:
                    st.metric("ROUGE-L", f"{eval_results['metrics']['avg_rougeL']:.3f}")
                
                # Display detailed results directly without nested expander
                st.markdown("### Detailed Results")
                for idx, result in enumerate(eval_results['detailed_results'], 1):
                    st.markdown(f"**Query {idx}:** {result['question']}")
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("Generated SQL:")
                        st.code(result['generated_sql'], language='sql')
                    with cols[1]:
                        st.markdown("Expected SQL:")
                        st.code(result['expected_sql'], language='sql')
                    st.markdown(f"ROUGE Scores: R1={result['rouge1']:.3f}, R2={result['rouge2']:.3f}, RL={result['rougeL']:.3f}")
                    st.markdown("---")
                    
            except FileNotFoundError:
                st.info("No evaluation results found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sql_query"):
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(message["sql_query"], language="sql")
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        for key, value in source.items():
                            st.markdown(f"- {key}: {value}")

    # Sidebar with example questions
    with st.sidebar:
        st.markdown("### 📝 Example Questions")
        sql_questions = [
            "How many incidents were reported in 2021?",
            "What are the top 5 most common hazards?",
            "Which country has the most incidents?",
            "What types of products were recalled in 2020?",
            "Show me the monthly trend of recalls in 2021"
        ]
        
        rag_questions = [
            "Describe the recall process for Listeria cases in meat products",
            "What are typical characteristics of plastic contamination incidents?",
            "Tell me about E. coli incidents in ground beef",
            "How are Class 1 recalls different from Class 2?",
            "Explain the handling of allergen-related recalls"
        ]
        
        st.markdown("#### Statistical Questions")
        for question in sql_questions:
            if st.button(question, key=f"sql_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
                
        st.markdown("#### Descriptive Questions")
        for question in rag_questions:
            if st.button(question, key=f"rag_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about food safety incidents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Process the current question if it exists
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing question type..."):
                try:
                    result = st.session_state.analyzer.process_query(
                        st.session_state.messages[-1]["content"]
                    )
                    
                    if result['success']:
                        # Handle different types of responses
                        if result['type'] == 'sql':
                            if result.get('sql_query'):
                                with st.expander("🔍 SQL Query", expanded=False):
                                    st.code(result['sql_query'], language="sql")
                            
                            formatted_answer = format_answer(result['answer'])
                            st.markdown(formatted_answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": formatted_answer,
                                "sql_query": result.get('sql_query')
                            })
                            
                        elif result['type'] == 'rag':
                            formatted_answer = format_rag_answer(result)
                            st.markdown(formatted_answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": formatted_answer,
                                "sources": result.get('sources')
                            })
                            
                        else:  # type == 'none'
                            st.markdown(result['answer'])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result['answer']
                            })
                    else:
                        st.error(result.get('message', 'An error occurred'))
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 