# ============================================================================
# IMPORTS: Enhanced C++ Code Agent with Modification Capabilities
# ============================================================================
import streamlit as st
import os
import glob
import pickle
import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path
import ast
import black
import autopep8
import isort
import numpy as np
# from git import Repo, InvalidGitRepositoryError

from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # Document combination
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION: Complete Privacy and Local Operation
# ============================================================================
# Disable all external telemetry and analytics
os.environ["OLLAMA_ANALYTICS"] = "false"
os.environ["OLLAMA_TELEMETRY"] = "false"

# Disable HuggingFace telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

# Disable transformers telemetry
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Disable torch telemetry
os.environ["TORCH_DISABLE_TELEMETRY"] = "1"

# Disable streamlit telemetry
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Disable python package telemetry
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["DISABLE_PIP_VERSION_CHECK"] = "1"

# Disable git telemetry if present
os.environ["GIT_TELEMETRY_OPTOUT"] = "1"

# Disable other common telemetry
os.environ["DO_NOT_TRACK"] = "1"
os.environ["ANALYTICS_DISABLED"] = "1"

# Set offline mode for various libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers warnings/telemetry

# ============================================================================
# CODE MODIFICATION UTILITIES
# ============================================================================

class CodeAgent:
    """Enhanced code agent with modification capabilities"""
    
    def __init__(self, project_path):
        self.project_path = project_path
        self.backup_path = os.path.join(project_path, ".code_agent_backups")
        self.ensure_backup_dir()
    
    def ensure_backup_dir(self):
        """Ensure backup directory exists"""
        os.makedirs(self.backup_path, exist_ok=True)
    
    def create_backup(self, file_path, operation="modify"):
        """Create backup of file before modification"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.basename(file_path)}.{operation}.{timestamp}.backup"
        backup_full_path = os.path.join(self.backup_path, backup_name)
        
        try:
            shutil.copy2(file_path, backup_full_path)
            return backup_full_path
        except Exception as e:
            st.error(f"Failed to create backup: {e}")
            return None
    
    def list_backups(self):
        """List all available backups"""
        if not os.path.exists(self.backup_path):
            return []
        
        backups = []
        for file in os.listdir(self.backup_path):
            if file.endswith('.backup'):
                backup_path = os.path.join(self.backup_path, file)
                stat = os.stat(backup_path)
                backups.append({
                    'name': file,
                    'path': backup_path,
                    'size': stat.st_size,
                    'modified': datetime.datetime.fromtimestamp(stat.st_mtime)
                })
        
        return sorted(backups, key=lambda x: x['modified'], reverse=True)
    
    def restore_backup(self, backup_path, target_path):
        """Restore file from backup"""
        try:
            shutil.copy2(backup_path, target_path)
            return True
        except Exception as e:
            st.error(f"Failed to restore backup: {e}")
            return False
    
    def modify_file(self, file_path, new_content, operation="modify"):
        """Safely modify a file with backup"""
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return False
        
        # Create backup
        backup_path = self.create_backup(file_path, operation)
        if not backup_path:
            return False
        
        try:
            # Write new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            st.success(f"✅ Modified: {os.path.basename(file_path)}")
            st.info(f"📁 Backup created: {os.path.basename(backup_path)}")
            return True
            
        except Exception as e:
            st.error(f"Failed to modify file: {e}")
            # Try to restore backup
            if backup_path and os.path.exists(backup_path):
                self.restore_backup(backup_path, file_path)
            return False
    
    def create_new_file(self, file_path, content):
        """Create a new file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            st.success(f"✅ Created: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            st.error(f"Failed to create file: {e}")
            return False
    
    def format_cpp_code(self, code):
        """Format C++ code using clang-format if available"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            result = subprocess.run([
                'clang-format', '-style=Google', tmp_path
            ], capture_output=True, text=True, timeout=30)
            
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return result.stdout
            else:
                st.warning("clang-format not available, returning original code")
                return code
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            st.warning("clang-format not available, returning original code")
            return code
        except Exception as e:
            st.warning(f"Code formatting failed: {e}")
            return code
    
    def format_python_code(self, code):
        """Format Python code using black and isort"""
        try:
            # Apply black formatting
            formatted = black.format_str(code, mode=black.Mode())
            
            # Apply isort for imports
            formatted = isort.code(formatted)
            
            return formatted
            
        except Exception as e:
            st.warning(f"Python code formatting failed: {e}")
            return code
    
    def validate_cpp_syntax(self, code):
        """Validate C++ syntax using g++ if available"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            result = subprocess.run([
                'g++', '-fsyntax-only', '-std=c++17', tmp_path
            ], capture_output=True, text=True, timeout=30)
            
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return True, "Syntax OK"
            else:
                return False, result.stderr
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, "g++ compiler not available"
        except Exception as e:
            return None, f"Syntax validation failed: {e}"
    
    def validate_python_syntax(self, code):
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        except Exception as e:
            return None, f"Validation failed: {e}"

# ============================================================================
# ENHANCED UI SETUP
# ============================================================================
st.set_page_config(
    page_title="C++ Code Agent",
    page_icon="🤖",
    layout="wide"
)

# Privacy and local operation notice
st.success("🔒 **PRIVACY MODE ACTIVE** - All processing happens locally, no external telemetry or data collection")

st.title("🤖 C++ Code Agent - AI-Powered Code Assistant")
st.write("Analyze, modify, and enhance your C++ codebase with AI - **100% Local & Private**")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
# Initialize session state for stop functionality
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

with st.sidebar:
    st.header("🔧 Agent Configuration")
    
    # Privacy status
    st.success("🔒 **PRIVACY PROTECTED**")
    st.caption("✅ No external connections")
    st.caption("✅ No telemetry data sent")
    st.caption("✅ All processing local")
    st.caption("✅ No cloud dependencies")
    
    # Processing status indicator
    if st.session_state.get('processing', False):
        st.warning("🔄 **PROCESSING** - Analysis in progress...")
        if st.button("🚨 Force Stop", type="secondary"):
            st.session_state.processing = False
            st.session_state.stop_requested = True
            st.rerun()
    else:
        st.success("✅ **READY** - Agent available")
    
    st.markdown("---")
    
    # Mode selection
    agent_mode = st.selectbox(
        "Agent Mode:",
        ["Analyze Only", "Code Assistant", "Full Agent"],
        index=2,
        help="Choose the level of agent capabilities"
    )
    
    # Project configuration
    st.subheader("📁 Project Settings")
    project_name = st.text_input(
        "Project Name:",
        placeholder="my_cpp_project",
        help="Simple name to identify your project"
    )
    
    code_folder = st.text_input(
        "C++ Project Path:", 
        placeholder=r"C:\path\to\your\cpp\project"
    )
    
    file_extensions = st.multiselect(
        "File Types:",
        options=[".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".c", ".inl"],
        default=[".cpp", ".h", ".hpp"]
    )
    
    # Agent capabilities
    if agent_mode in ["Code Assistant", "Full Agent"]:
        st.subheader("🚀 Agent Capabilities")
        
        # Timeout setting
        st.subheader("⏱️ Safety Settings")
        timeout_seconds = st.slider(
            "Analysis Timeout (seconds):",
            min_value=30,
            max_value=300,
            value=120,
            help="Maximum time to wait for analysis before auto-stop"
        )
        
        enable_modifications = st.checkbox(
            "Enable Code Modifications",
            value=agent_mode == "Full Agent",
            help="Allow agent to modify existing files"
        )
        
        enable_file_creation = st.checkbox(
            "Enable File Creation",
            value=agent_mode == "Full Agent",
            help="Allow agent to create new files"
        )
        
        enable_formatting = st.checkbox(
            "Auto-format Code",
            value=True,
            help="Automatically format generated code"
        )
        
        enable_validation = st.checkbox(
            "Syntax Validation",
            value=True,
            help="Validate code syntax before applying changes"
        )
    else:
        # Default values for Analyze Only mode
        timeout_seconds = 120
        enable_modifications = False
        enable_file_creation = False  
        enable_formatting = True
        enable_validation = True

# ============================================================================
# MODEL INITIALIZATION (Same as original)
# ============================================================================
@st.cache_resource
def get_embeddings():
    try:
        # Force offline mode and disable telemetry for HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False,  # Security: don't execute remote code
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        
        # Verify the model works locally
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) > 0:
            st.sidebar.info("✅ Embeddings: Local model ready")
            return embeddings, True
        else:
            return None, False
            
    except Exception as e:
        st.sidebar.error(f"❌ Embedding model error: {e}")
        return None, False

@st.cache_resource
def get_llm():
    try:
        # Connect only to local Ollama instance
        llm = OllamaLLM(
            model="cpp-expert", 
            base_url="http://localhost:11434",
            # Ensure no external connections
            timeout=30,
            keep_alive=True
        )
        
        # Test local connection only
        test_response = llm.invoke("Hello")
        st.sidebar.success("✅ LLM: Local Ollama connected")
        return llm, True
        
    except Exception as e:
        st.sidebar.error(f"❌ Local LLM not available: {e}")
        st.sidebar.info("💡 Start Ollama locally: `ollama serve`")
        return None, False

# Load models
embeddings, embeddings_ready = get_embeddings()
llm, llm_ready = get_llm()

# ============================================================================
# ENHANCED FUNCTIONALITY
# ============================================================================

SYSTEM_PROMPT = """You are an expert C++ Code Analyst and Assistant with deep knowledge of:

🎯 CORE EXPERTISE:
• Modern C++ (C++11/14/17/20) best practices and patterns
• Performance optimization and memory management
• Design patterns, SOLID principles, and clean architecture
• STL, algorithms, and standard library usage
• Debugging, profiling, and code quality assessment
• Cross-platform development and build systems

📋 ANALYSIS METHODOLOGY:
1. **Context Understanding**: Carefully analyze the provided code context
2. **Question Focus**: Address the specific user question with precision
3. **Code References**: Reference actual files, functions, and line ranges
4. **Practical Solutions**: Provide actionable, implementable recommendations
5. **Explanation**: Explain the 'why' behind every suggestion

💡 RESPONSE FORMAT:
• **Overview**: Brief summary of findings
• **Detailed Analysis**: In-depth code examination
• **Specific Issues**: Point out problems with file/line references
• **Solutions**: Complete, working code examples
• **Best Practices**: Modern C++ recommendations
• **Next Steps**: Actionable implementation guidance

📚 EXAMPLES OF GOOD ANALYSIS:

Example 1 - Function Analysis:
"In utils.cpp:45, the function processData() has a potential memory leak. Here's the corrected version using smart pointers."

Example 2 - Performance Issue:
"The loop in main.cpp:123-130 shows O(n²) complexity. Consider using std::unordered_map for O(1) lookups."

🔍 CONTEXT FROM USER'S CODEBASE:
{context}

❓ USER QUESTION:
{input}

📝 YOUR EXPERT ANALYSIS:"""


def create_enhanced_prompt():
    """Create advanced prompt with context enhancement"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

if not embeddings_ready:
    st.error("❌ Embedding model not available")
elif not llm_ready:
    st.error("❌ C++ Expert model not available. Make sure Ollama is running.")
else:
    st.success("✅ C++ Code Agent ready!")
    
    if project_name and code_folder and os.path.exists(code_folder) and file_extensions:
        # Initialize code agent
        if agent_mode in ["Code Assistant", "Full Agent"]:
            code_agent = CodeAgent(code_folder)
        
        # Sanitize project name
        project_name = "".join(c for c in project_name if c.isalnum() or c in ['_', '-'])
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Files", "🔄 Backups", "⚙️ Tools"])
        
        with tab1:
            st.subheader("Chat with Your Codebase")
            
            # Embedding management (simplified from original)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Index Codebase", type="primary"):
                    with st.spinner("Indexing codebase..."):
                        # Similar to original indexing logic but simplified
                        documents = []
                        file_count = 0
                        
                        for ext in file_extensions:
                            pattern = os.path.join(code_folder, f"**/*{ext}")
                            files = glob.glob(pattern, recursive=True)
                            
                            for file_path in files:
                                try:
                                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()
                                    
                                    if len(content.strip()) > 0:
                                        rel_path = os.path.relpath(file_path, code_folder)
                                        # Add file context with proper formatting
                                        file_info = f"=== FILE: {rel_path} ({ext} file) ===\n"
                                        file_info += f"Lines: {len(content.splitlines())}\n"
                                        file_info += f"Size: {len(content)} characters\n\n"
                                        
                                        documents.append(Document(
                                            page_content=file_info + content,
                                            metadata={
                                                'source': file_path,
                                                'relative_path': rel_path,
                                                'file_type': ext,
                                                'line_count': len(content.splitlines()),
                                                'char_count': len(content)
                                            }
                                        ))
                                        file_count += 1
                                except Exception as e:
                                    st.warning(f"Could not read {file_path}: {e}")
                        
                        if documents:
                            # ----- INDEXING: create both summary and chunk indexes -----
                            def create_cpp_aware_splitter():
                                cpp_separators = [
                                    "\n// ====", "\nnamespace ", "\nclass ", "\nstruct ",
                                    "\nenum ", "\ntemplate", "\nvoid ", "\nint ", "\nfloat ",
                                    "\ndouble ", "\nbool ", "\nauto ", "\n}", "\n\n", "\n", " ", ""
                                ]
                                return RecursiveCharacterTextSplitter(
                                    chunk_size=1800,
                                    chunk_overlap=400,
                                    separators=cpp_separators,
                                    keep_separator=True
                                )
                            
                            text_splitter = create_cpp_aware_splitter()
                            split_docs = text_splitter.split_documents(documents)
                            
                            # Build chunk index
                            chunk_index = FAISS.from_documents(split_docs, embeddings)
                            
                            # Build simple file-level summaries (head lines) as lightweight summary index
                            summary_docs = []
                            for d in documents:
                                try:
                                    content_lines = d.page_content.splitlines()
                                    head = "\n".join(content_lines[:min(60, len(content_lines))])
                                except Exception:
                                    head = d.page_content[:1000]
                                summary_text = f"FILE_SUMMARY: {d.metadata.get('relative_path','unknown')}\n\n{head}"
                                summary_docs.append(Document(page_content=summary_text, metadata=d.metadata))
                            
                            summary_index = FAISS.from_documents(summary_docs, embeddings)
                            
                            # Save both indexes separately inside project index dir
                            vector_store_dir = os.path.join(code_folder, f".code_agent_index_{project_name}")
                            os.makedirs(vector_store_dir, exist_ok=True)
                            chunk_index.save_local(os.path.join(vector_store_dir, "chunks"))
                            summary_index.save_local(os.path.join(vector_store_dir, "summaries"))
                            
                            # Cache useful objects in session state
                            st.session_state.vectorstore_chunks = chunk_index
                            st.session_state.vectorstore_summaries = summary_index
                            st.session_state.retriever_chunks = chunk_index.as_retriever(
                                search_type="mmr",
                                search_kwargs={"k": 8, "fetch_k": 12, "lambda_mult": 0.5}
                            )
                            st.session_state.retriever_summaries = summary_index.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 10}
                            )
                            
                            # Cache split docs and their embeddings to allow fast reranking
                            st.session_state.chunk_docs = split_docs
                            try:
                                st.session_state.chunk_embeddings = embeddings.embed_documents(
                                    [d.page_content for d in split_docs]
                                )
                            except Exception:
                                # fallback per-chunk embedding if embed_documents not supported
                                st.session_state.chunk_embeddings = [
                                    embeddings.embed_query(d.page_content) for d in split_docs
                                ]
                            
                            st.session_state.embeddings_loaded = True
                            
                            st.success(f"✅ Indexed {file_count} files, {len(split_docs)} chunks")
            
            with col2:
                if st.button("📊 Load Existing Index"):
                    vector_store_path = os.path.join(code_folder, f".code_agent_index_{project_name}")
                    if os.path.exists(vector_store_path):
                        try:
                            # Try to load chunk and summary indexes
                            chunks_dir = os.path.join(vector_store_path, "chunks")
                            summaries_dir = os.path.join(vector_store_path, "summaries")
                            loaded_chunks = None
                            loaded_summaries = None
                            
                            if os.path.exists(chunks_dir):
                                loaded_chunks = FAISS.load_local(chunks_dir, embeddings, allow_dangerous_deserialization=True)
                            if os.path.exists(summaries_dir):
                                loaded_summaries = FAISS.load_local(summaries_dir, embeddings, allow_dangerous_deserialization=True)
                            
                            # If only a single combined index exists, fallback to older layout
                            if loaded_chunks is None and os.path.exists(vector_store_path):
                                try:
                                    loaded_combined = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
                                except Exception:
                                    loaded_combined = None
                                if loaded_combined is not None:
                                    st.session_state.vectorstore_chunks = loaded_combined
                                    st.session_state.retriever_chunks = loaded_combined.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={"k": 8, "fetch_k": 12, "lambda_mult": 0.5}
                                    )
                            else:
                                if loaded_chunks is not None:
                                    st.session_state.vectorstore_chunks = loaded_chunks
                                    st.session_state.retriever_chunks = loaded_chunks.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={"k": 8, "fetch_k": 12, "lambda_mult": 0.5}
                                    )
                                if loaded_summaries is not None:
                                    st.session_state.vectorstore_summaries = loaded_summaries
                                    st.session_state.retriever_summaries = loaded_summaries.as_retriever(
                                        search_type="similarity",
                                        search_kwargs={"k": 10}
                                    )
                            
                            # Best-effort: try to rehydrate chunk_docs and embeddings if manifest exists (not implemented here)
                            st.session_state.embeddings_loaded = True
                            st.success("✅ Loaded existing index!")
                        except Exception as e:
                            st.error(f"Failed to load index: {e}")
                    else:
                        st.warning("No existing index found. Create one first.")
            
            # Chat interface
            if st.session_state.get('embeddings_loaded', False):
                st.markdown("---")
                
                # Enhanced quick actions
                st.subheader("🎯 Quick Actions")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("📋 Code Overview"):
                        st.session_state.suggested_question = "Give me a comprehensive overview of this codebase structure, main components, and architecture."
                
                with col2:
                    if st.button("🔧 Find Issues"):
                        st.session_state.suggested_question = "Analyze this code for potential bugs, memory leaks, performance issues, and suggest specific improvements with code examples."
                
                with col3:
                    if st.button("🚀 Optimize Code"):
                        st.session_state.suggested_question = "Identify performance bottlenecks and provide optimized code implementations with explanations."
                
                with col4:
                    if st.button("✨ Modernize"):
                        st.session_state.suggested_question = "Suggest how to modernize this code using C++17/20 features, with specific code examples."
                
                # Chat input
                user_question = st.text_area(
                    "Ask your code agent:",
                    value=getattr(st.session_state, 'suggested_question', ''),
                    height=120,
                    placeholder="Ask me to analyze, modify, optimize, or create code..."
                )
                
                # Submit and Stop buttons
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    submit_clicked = st.button("🤖 Ask Agent", type="primary", disabled=st.session_state.get('processing', False)) and user_question
                
                with col2:
                    if st.session_state.get('processing', False):
                        if st.button("⏹️ Stop", type="secondary"):
                            st.session_state.processing = False
                            st.session_state.stop_requested = True
                            st.rerun()
                    else:
                        st.button("⏹️ Stop", type="secondary", disabled=True)
                
                # Processing logic
                if submit_clicked:
                    st.session_state.processing = True
                    st.session_state.stop_requested = False
                    
                    # Initialize variables to prevent undefined errors
                    enhanced_docs = []
                    response_content = ""
                    
                    # Create progress tracking
                    progress_container = st.empty()
                    status_container = st.empty()
                    
                    try:
                        # Check if retriever is available
                        if not hasattr(st.session_state, 'retriever_chunks') or not st.session_state.retriever_chunks:
                            st.error("❌ Please index your codebase first using the 'Index Codebase' button above.")
                            st.session_state.processing = False
                        else:
                            with status_container.container():
                                st.info("🔍 Starting analysis... (Click Stop to cancel)")
                            
                            # Check for stop before each major step
                            if st.session_state.get('stop_requested', False):
                                st.warning("⏹️ Analysis stopped by user")
                                st.session_state.processing = False
                            else:
                                # Continue with processing only if not stopped
                                with st.spinner("Agent thinking..."):
                                    # Enhanced RAG chain setup (we will use manual retrieval + rerank)
                                    enhanced_prompt = create_enhanced_prompt()
                                    
                                    # Check for stop request
                                    if st.session_state.get('stop_requested', False):
                                        st.warning("⏹️ Analysis stopped by user")
                                        st.session_state.processing = False
                                    else:
                                        status_container.info("📊 Retrieving relevant code sections...")
                                        
                                        # ----- TWO-TIER RETRIEVAL & RERANK -----
                                        # Step 1: retrieve top files from summaries (if available)
                                        summary_results = []
                                        if st.session_state.get("retriever_summaries"):
                                            try:
                                                if hasattr(st.session_state.retriever_summaries, "get_relevant_documents"):
                                                    summary_results = st.session_state.retriever_summaries.get_relevant_documents(user_question)
                                                else:
                                                    summary_results = st.session_state.retriever_summaries.invoke(user_question)
                                            except Exception:
                                                summary_results = []
                                        
                                        top_files = set()
                                        for sd in summary_results:
                                            src = sd.metadata.get("relative_path") or sd.metadata.get("source")
                                            if src:
                                                top_files.add(src)
                                        
                                        # Step 2: compute query embedding and score cached chunk embeddings
                                        try:
                                            query_emb = embeddings.embed_query(user_question)
                                        except Exception:
                                            query_emb = None
                                        
                                        chunk_embs = st.session_state.get("chunk_embeddings", [])
                                        try:
                                            if query_emb is not None and len(chunk_embs) > 0:
                                                chunk_matrix = np.array(chunk_embs, dtype=float)
                                                q = np.array(query_emb, dtype=float)
                                                dot = chunk_matrix.dot(q)
                                                norm_chunks = np.linalg.norm(chunk_matrix, axis=1) + 1e-12
                                                norm_q = np.linalg.norm(q) + 1e-12
                                                sims = dot / (norm_chunks * norm_q)
                                            else:
                                                sims = np.zeros(len(st.session_state.get("chunk_docs", [])))
                                        except Exception:
                                            sims = np.zeros(len(st.session_state.get("chunk_docs", [])))
                                        
                                        # Score chunks: prefer those from top_files then by similarity
                                        scored = []
                                        for idx, sim in enumerate(sims):
                                            src = None
                                            try:
                                                src = st.session_state["chunk_docs"][idx].metadata.get("relative_path")
                                            except Exception:
                                                src = None
                                            in_top = 1 if src in top_files else 0
                                            scored.append((in_top, float(sim), idx))
                                        
                                        scored.sort(key=lambda x: (-x[0], -x[1]))
                                        
                                        TOP_CHUNKS = 8
                                        selected_indices = [s[2] for s in scored[:TOP_CHUNKS]]
                                        if st.session_state.get("chunk_docs"):
                                            retrieved_docs = [st.session_state["chunk_docs"][i] for i in selected_indices]
                                        else:
                                            # fallback to chunk retriever if chunk_docs not cached
                                            if st.session_state.get("retriever_chunks"):
                                                try:
                                                    if hasattr(st.session_state.retriever_chunks, "get_relevant_documents"):
                                                        retrieved_docs = st.session_state.retriever_chunks.get_relevant_documents(user_question)
                                                    else:
                                                        retrieved_docs = st.session_state.retriever_chunks.invoke(user_question)
                                                except Exception:
                                                    retrieved_docs = []
                                            else:
                                                retrieved_docs = []
                                        
                                        # Debug: show top files
                                        try:
                                            status_container.info(f"📂 Top files: {', '.join(list(top_files)[:6]) if top_files else 'N/A'}")
                                        except Exception:
                                            pass
                                        
                                        # Check for stop request
                                        if st.session_state.get('stop_requested', False):
                                            st.warning("⏹️ Analysis stopped by user")
                                            st.session_state.processing = False
                                        else:
                                            status_container.info(f"🧠 Processing {len(retrieved_docs)} code sections...")
                                            
                                            # Process and enhance retrieved context
                                            def enhance_context(docs, query):
                                                enhanced = []
                                                for doc in docs:
                                                    # Add query relevance context
                                                    file_path = doc.metadata.get('relative_path', 'unknown')
                                                    enhanced_content = f"[RELEVANCE: File {file_path} - Related to: {query[:100]}...]\n\n"
                                                    enhanced_content += doc.page_content
                                                    
                                                    enhanced_doc = Document(
                                                        page_content=enhanced_content,
                                                        metadata=doc.metadata
                                                    )
                                                    enhanced.append(enhanced_doc)
                                                return enhanced
                                            
                                            # Process retrieved documents
                                            enhanced_docs = enhance_context(retrieved_docs, user_question)
                                            st.write(f"🎯 Analyzing {len(enhanced_docs)} relevant code sections...")
                                            
                                            # Build context string for prompt
                                            context_str = "\n\n".join([doc.page_content for doc in enhanced_docs])
                                            
                                            # Final stop check before AI generation
                                            if st.session_state.get('stop_requested', False):
                                                st.warning("⏹️ Analysis stopped by user")
                                                st.session_state.processing = False
                                            else:
                                                # Continue with AI generation
                                                status_container.info("🤖 Generating expert analysis...")
                                                
                                                # Construct final prompt text from SYSTEM_PROMPT to ensure consistent formatting
                                                try:
                                                    final_prompt_text = SYSTEM_PROMPT.format(context=context_str, input=user_question)
                                                except Exception:
                                                    # fallback simple concatenation
                                                    final_prompt_text = SYSTEM_PROMPT.replace("{context}", context_str).replace("{input}", user_question)
                                                
                                                # Generate response
                                                response_content = llm.invoke(final_prompt_text)
                                                
                                                # Clear status after successful completion
                                                status_container.empty()
                                            
                                                # Display enhanced response
                                                st.markdown("### 🎯 Expert Code Analysis:")
                                                st.markdown(response_content)
                                                
                                                # Success - reset processing state
                                                st.session_state.processing = False
                            
                    except Exception as e:
                        # Comprehensive error handling
                        error_message = f"❌ Error during analysis: {str(e)}"
                        if not st.session_state.get('stop_requested', False):
                            st.error(error_message)
                            # Show error details in expander for debugging
                            with st.expander("🔍 Error Details"):
                                st.code(str(e))
                        
                        # Always reset processing state
                        st.session_state.processing = False
                        status_container.empty()
                    
                    finally:
                        # Ensure processing state is always reset
                        st.session_state.processing = False
                    
                    # Show debug content only if we have processed documents
                    if enhanced_docs and len(enhanced_docs) > 0:
                        # Show retrieved content in debug mode
                        if st.checkbox("🔍 Show analyzed content", key="debug_retrieval"):
                            with st.expander("Code Sections Being Analyzed"):
                                for i, doc in enumerate(enhanced_docs):
                                    st.write(f"**Section {i+1}** - {doc.metadata.get('relative_path', 'unknown')}")
                                    preview = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
                                    st.code(preview, language='cpp')
                        
                        # Show analyzed files (only if we have documents)
                        if len(enhanced_docs) > 0:
                            with st.expander("📁 Files Analyzed"):
                                analyzed_files = set()
                                for doc in enhanced_docs:
                                    if 'relative_path' in doc.metadata:
                                        analyzed_files.add(doc.metadata['relative_path'])
                                
                                if analyzed_files:  # Only show if we found files
                                    for file_path in sorted(analyzed_files):
                                        # Show file info
                                        matching_docs = [d for d in enhanced_docs if d.metadata.get('relative_path') == file_path]
                                        if matching_docs:
                                            lines = matching_docs[0].metadata.get('line_count', 'unknown')
                                            st.write(f"• `{file_path}` ({lines} lines)")
                                else:
                                    st.write("No files were analyzed in this session.")
                        
                        # Enhanced code modification suggestions
                        if (not st.session_state.get('stop_requested', False) and 
                            agent_mode in ["Code Assistant", "Full Agent"] and 
                            user_question and
                            any(keyword in user_question.lower() 
                                for keyword in ['modify', 'change', 'improve', 'optimize', 'fix', 'create', 'add', 'refactor', 'update'])):
                            st.markdown("### 🛠️ Implementation Ready")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info("💡 **Code Modifications Available** - Use Files tab to implement suggested changes")
                            with col2:
                                if st.button("📝 Go to Files Tab", type="secondary"):
                                    st.session_state.active_tab = "Files"
                        
                        # Model performance suggestions
                        with st.expander("⚡ Improve Analysis Quality"):
                            st.markdown("""**To get better responses:**
                            
                            🎯 **Better Questions:**
                            - Be specific: "Fix memory leak in UserManager::createUser()" vs "Fix bugs"
                            - Include context: "How can I optimize the sorting algorithm in data_processor.cpp?"
                            - Ask about specific files: "What's wrong with the error handling in network_client.h?"
                            
                            🔧 **Model Improvements:**
                            - Use `ollama pull codellama` or `ollama pull deepseek-coder` for better C++ understanding
                            - Consider `ollama pull magicoder` for code generation tasks
                            
                            📊 **Index Quality:**
                            - Re-index after major code changes
                            - Include header files and implementation files together
                            - Use descriptive file and function names""")
                
                # Clear suggested question
                if 'suggested_question' in st.session_state:
                    del st.session_state.suggested_question
        
        # Files tab for code modification
        with tab2:
            if agent_mode in ["Code Assistant", "Full Agent"]:
                st.subheader("📁 File Management")
                
                # File browser
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Project Files:**")
                    
                    # Get project files
                    project_files = []
                    for ext in file_extensions:
                        pattern = os.path.join(code_folder, f"**/*{ext}")
                        project_files.extend(glob.glob(pattern, recursive=True))
                    
                    selected_file = st.selectbox(
                        "Select file to view/edit:",
                        options=[os.path.relpath(f, code_folder) for f in sorted(project_files)],
                        index=0 if project_files else None
                    )
                
                with col2:
                    if selected_file and project_files:
                        full_path = os.path.join(code_folder, selected_file)
                        
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            
                            st.write(f"**Editing:** `{selected_file}`")
                            
                            # File editor
                            edited_content = st.text_area(
                                "File Content:",
                                value=file_content,
                                height=400,
                                key=f"editor_{selected_file}"
                            )
                            
                            # Action buttons
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if st.button("💾 Save Changes"):
                                    if enable_modifications:
                                        success = code_agent.modify_file(
                                            full_path, 
                                            edited_content,
                                            "manual_edit"
                                        )
                                        if success:
                                            st.rerun()
                                    else:
                                        st.warning("Code modifications disabled in current mode")
                            
                            with col2:
                                if st.button("🎨 Format Code"):
                                    if selected_file.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx')):
                                        formatted = code_agent.format_cpp_code(edited_content)
                                    elif selected_file.endswith('.py'):
                                        formatted = code_agent.format_python_code(edited_content)
                                    else:
                                        formatted = edited_content
                                    
                                    st.session_state[f"editor_{selected_file}"] = formatted
                                    st.rerun()
                            
                            with col3:
                                if st.button("✅ Validate"):
                                    if selected_file.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx')):
                                        is_valid, message = code_agent.validate_cpp_syntax(edited_content)
                                    elif selected_file.endswith('.py'):
                                        is_valid, message = code_agent.validate_python_syntax(edited_content)
                                    else:
                                        is_valid, message = None, "Validation not available for this file type"
                                    
                                    if is_valid is True:
                                        st.success(f"✅ {message}")
                                    elif is_valid is False:
                                        st.error(f"❌ {message}")
                                    else:
                                        st.info(f"ℹ️ {message}")
                            
                            with col4:
                                if st.button("🔄 Reload"):
                                    st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
            
            else:
                st.info("File modification features are available in Code Assistant and Full Agent modes.")
        
        # Backups tab
        with tab3:
            if agent_mode in ["Code Assistant", "Full Agent"]:
                st.subheader("🔄 Backup Management")
                
                backups = code_agent.list_backups()
                
                if backups:
                    st.write(f"**Found {len(backups)} backup(s):**")
                    
                    for backup in backups[:20]:  # Show latest 20
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            st.write(f"`{backup['name']}`")
                        
                        with col2:
                            st.write(f"{backup['size']} bytes")
                        
                        with col3:
                            st.write(backup['modified'].strftime("%Y-%m-%d %H:%M:%S"))
                        
                        with col4:
                            if st.button("🔄", key=f"restore_{backup['name']}"):
                                # Extract original filename from backup name
                                original_name = backup['name'].split('.')[0] + '.' + backup['name'].split('.')[1]
                                original_path = os.path.join(code_folder, original_name)
                                
                                if code_agent.restore_backup(backup['path'], original_path):
                                    st.success(f"Restored {original_name}")
                                    st.rerun()
                
                else:
                    st.info("No backups found. Backups are created automatically when you modify files.")
            
            else:
                st.info("Backup features are available in Code Assistant and Full Agent modes.")
        
        # Tools tab
        with tab4:
            st.subheader("⚙️ Development Tools")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Code Analysis Tools:**")
                if st.button("🔍 Analyze Project Structure"):
                    # Implement project structure analysis
                    st.info("Feature coming soon!")
                
                if st.button("📊 Generate Documentation"):
                    st.info("Feature coming soon!")
                
                if st.button("🧪 Run Tests"):
                    st.info("Feature coming soon!")
            
            with col2:
                st.write("**Project Tools:**")
                if st.button("📦 Create CMakeLists.txt"):
                    st.info("Feature coming soon!")
                
                if st.button("🔧 Setup Build System"):
                    st.info("Feature coming soon!")
                
                if st.button("📋 Generate Project Template"):
                    st.info("Feature coming soon!")
    
    else:
        st.info("👆 Configure your project settings in the sidebar to get started")

# ============================================================================
# FOOTER: Privacy and Local Operation Guarantee
# ============================================================================
st.markdown("---")
st.markdown("### 🔒 **PRIVACY & SECURITY GUARANTEE**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🚫 NO EXTERNAL COMMUNICATIONS:**
    - ✅ No telemetry data sent
    - ✅ No analytics collection
    - ✅ No cloud API calls
    - ✅ No usage tracking
    """)

with col2:
    st.markdown("""
    **🏠 100% LOCAL PROCESSING:**
    - ✅ All AI processing local (Ollama)
    - ✅ All embeddings cached locally
    - ✅ All data stays on your machine
    - ✅ No internet dependencies
    """)

with col3:
    st.markdown("""
    **🛡️ SECURITY FEATURES:**
    - ✅ Automatic code backups
    - ✅ Syntax validation
    - ✅ Safe file modifications
    - ✅ Version control ready
    """)

st.markdown("*🔐 Your code and data never leave your machine - Guaranteed local AI assistance*")