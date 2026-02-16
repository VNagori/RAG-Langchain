# ============================================================================
# IMPORTS: Import all required libraries for the C++ Code Expert application
# ============================================================================
import streamlit as st                                    # Web UI framework for creating the interface
from langchain_ollama import OllamaLLM                    # Interface to connect to local Ollama LLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # File loading utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter          # Splits large documents into chunks
from langchain_huggingface import HuggingFaceEmbeddings   # Text embedding model from HuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder   # Prompt template creation
from langchain_classic.chains import create_retrieval_chain               # RAG chain for document retrieval
from langchain_classic.chains import create_history_aware_retriever       # Context-aware retrieval
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # Document combination
from langchain_core.runnables.history import RunnableWithMessageHistory   # Chat history management
from langchain_community.chat_message_histories import ChatMessageHistory  # Store chat messages
from langchain_core.chat_history import BaseChatMessageHistory            # Base class for chat history
from langchain_core.documents import Document             # Document object for storing text and metadata
import os                                                 # Operating system interface
import glob                                               # File pattern matching
import pickle                                             # Python object serialization
import datetime                                           # For timestamp generation
import shutil                                             # File and directory operations

# ============================================================================
# CONFIGURATION: Setup environment and disable telemetry
# ============================================================================
# Disable Ollama analytics/telemetry to ensure complete privacy
os.environ["OLLAMA_ANALYTICS"] = "false"

# Disable Chroma telemetry and set safe database mode
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"

# Set SQLite to use safer settings
os.environ["SQLITE_TMPDIR"] = os.path.join(os.path.dirname(__file__), "temp")
# Ensure temp directory exists
temp_dir = os.environ["SQLITE_TMPDIR"]
os.makedirs(temp_dir, exist_ok=True)

# ============================================================================
# STREAMLIT UI SETUP: Create the main application interface
# ============================================================================
st.title("🚀 C++ Code Expert - Local Codebase Assistant")
st.write("Analyze your C++ codebase with intelligent embedding management!")

# ============================================================================
# EMBEDDING MODEL INITIALIZATION: Setup HuggingFace embeddings with caching
# ============================================================================
@st.cache_resource  # Cache the embeddings model to avoid reloading on every run
def get_embeddings():
    """
    Initialize HuggingFace embeddings model for converting text to vector representations.
    This model converts code text into numerical vectors for similarity search.
    
    Returns:
        tuple: (embeddings_model, success_flag)
    """
    try:
        # Initialize the sentence transformer model for creating embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight, fast embedding model
            model_kwargs={'device': 'cpu'},                       # Force CPU usage (works on any machine)
            encode_kwargs={'normalize_embeddings': True}          # Normalize vectors for better similarity comparison
        )
        return embeddings, True
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, False

# ============================================================================
# LLM INITIALIZATION: Setup local Ollama LLM with caching
# ============================================================================
@st.cache_resource  # Cache the LLM to avoid reconnection on every run
def get_llm():
    """
    Initialize connection to local Ollama LLM (cpp-expert model).
    This connects to your locally running C++ expert model.
    
    Returns:
        tuple: (llm_instance, success_flag)
    """
    try:
        # Connect to local Ollama server running the cpp-expert model
        llm = OllamaLLM(model="cpp-expert", base_url="http://localhost:11434")
        test_response = llm.invoke("Hello")  # Test connection with simple query
        return llm, True
    except Exception as e:
        return None, False

# ============================================================================
# SIMPLIFIED CACHE MANAGEMENT: Simple embedding persistence without hashing
# ============================================================================
def save_embedding_metadata(project_name, metadata):
    """
    Save metadata about created embeddings to disk using project name.
    
    Args:
        project_name (str): Simple name for the project
        metadata (dict): Information about the embedding process
        
    Returns:
        str: Path to the saved metadata file
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), "embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use simple project name for filename
    cache_file = os.path.join(cache_dir, f"embedding_meta_{project_name}.pkl")
    
    # Save metadata using pickle serialization
    with open(cache_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    return cache_file

def load_embedding_metadata(project_name):
    """
    Load previously saved embedding metadata from disk.
    
    Args:
        project_name (str): Name of the project
        
    Returns:
        dict or None: Metadata if found, None otherwise
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "embedding_cache")
    cache_file = os.path.join(cache_dir, f"embedding_meta_{project_name}.pkl")
    
    # Try to load existing metadata
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None  # Return None if file is corrupted
    return None

def check_embedding_exists(project_name):
    """
    Check if valid embeddings exist for the specified project.
    
    Args:
        project_name (str): Name of the project to check
        
    Returns:
        tuple: (exists_flag, metadata_dict)
    """
    metadata = load_embedding_metadata(project_name)
    if metadata:
        # Verify that the actual vector store files still exist on disk
        vector_store_path = metadata.get('vector_store_path')
        if vector_store_path and os.path.exists(vector_store_path):
            return True, metadata
    return False, None

def cleanup_chroma_instances():
    """
    Clean up any existing Chroma instances to prevent conflicts.
    """
    try:
        import gc
        
        # Clear session state variables that might hold Chroma instances
        if 'retriever' in st.session_state:
            del st.session_state.retriever
        if 'embeddings_loaded' in st.session_state:
            del st.session_state.embeddings_loaded
            
        # Force garbage collection
        gc.collect()
        
    except Exception:
        # If cleanup fails, continue silently
        pass

def cleanup_database_files(vector_store_path):
    """
    Safely cleanup corrupted or locked database files.
    
    Args:
        vector_store_path (str): Path to the vector store directory
    """
    try:
        if os.path.exists(vector_store_path):
            # Try to remove the directory and all its contents
            shutil.rmtree(vector_store_path, ignore_errors=True)
            # Wait a moment and ensure it's completely gone
            import time
            time.sleep(0.5)
            if os.path.exists(vector_store_path):
                # If still exists, try removing individual files
                for root, dirs, files in os.walk(vector_store_path, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except:
                            pass
                try:
                    os.rmdir(vector_store_path)
                except:
                    pass
    except Exception as e:
        st.warning(f"Warning: Could not fully cleanup database files: {e}")

def ensure_safe_directory(directory_path):
    """
    Safely create directory with proper permissions.
    
    Args:
        directory_path (str): Path to create
        
    Returns:
        bool: True if directory is ready for use
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        
        # Test if we can write to the directory
        test_file = os.path.join(directory_path, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except:
            return False
            
    except Exception:
        return False

def delete_embedding_data(project_name):
    """
    Delete embedding data and metadata for a project.
    
    Args:
        project_name (str): Name of the project to delete
        
    Returns:
        bool: True if deletion was successful
    """
    try:
        # Delete vector store
        metadata = load_embedding_metadata(project_name)
        if metadata:
            vector_store_path = metadata.get('vector_store_path')
            if vector_store_path:
                cleanup_database_files(vector_store_path)
        
        # Delete metadata file
        cache_dir = os.path.join(os.path.dirname(__file__), "embedding_cache")
        cache_file = os.path.join(cache_dir, f"embedding_meta_{project_name}.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        return True
    except Exception as e:
        st.error(f"Error deleting embeddings: {e}")
        return False

# ============================================================================
# HISTORY MANAGEMENT FUNCTIONS: Handle chat history size and memory optimization
# ============================================================================
def get_history_size(session_history):
    """
    Calculate the approximate size of chat history in memory.
    
    Args:
        session_history: ChatMessageHistory object
        
    Returns:
        tuple: (message_count, estimated_size_bytes)
    """
    if not session_history or not session_history.messages:
        return 0, 0
    
    total_chars = 0
    for message in session_history.messages:
        total_chars += len(str(message.content))
    
    # Rough estimation: 1 char = 1 byte + overhead
    estimated_size = total_chars * 2  # Account for Python string overhead
    return len(session_history.messages), estimated_size

def format_size(size_bytes):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

# ============================================================================
# MODEL INITIALIZATION: Load both embeddings and LLM models
# ============================================================================
# Initialize the embedding model (downloads once, then cached locally)
embeddings, embeddings_ready = get_embeddings()
# Initialize the local LLM connection
llm, llm_ready = get_llm()

# ============================================================================
# ERROR HANDLING: Check if both models are ready before proceeding
# ============================================================================
if not embeddings_ready:
    st.error("❌ Embedding model not available")
elif not llm_ready:
    st.error("❌ C++ Expert model not available. Make sure Ollama is running and cpp-expert model exists.")
else:
    st.success("✅ C++ Code Expert ready!")
    
    # ========================================================================
    # PROJECT CONFIGURATION: Get project settings
    # ========================================================================
    # Project name for identifying embeddings
    project_name = st.text_input(
        "📋 Project Name:",
        placeholder="my_cpp_project",
        help="Simple name to identify your embeddings (no spaces or special characters)"
    )
    
    # Input field for user to specify their C++ project folder
    code_folder = st.text_input(
        "📁 Enter path to your C++ code folder:", 
        placeholder=r"C:\path\to\your\cpp\project"
    )
    
    # Multi-select dropdown for choosing which C++ file types to analyze
    file_extensions = st.multiselect(
        "🔧 Select file types to analyze:",
        options=[".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".c", ".inl"],
        default=[".cpp", ".h", ".hpp"]  # Common C++ file extensions pre-selected
    )
    
    # ========================================================================
    # MAIN PROCESSING SECTION: Only proceed if valid inputs
    # ========================================================================
    if project_name and code_folder and os.path.exists(code_folder) and file_extensions:
        # Sanitize project name (remove special characters)
        project_name = "".join(c for c in project_name if c.isalnum() or c in ['_', '-'])
        
        # Check if embeddings already exist for this project
        embedding_exists, existing_metadata = check_embedding_exists(project_name)
        
        # ====================================================================
        # EMBEDDING STATUS DISPLAY: Show current embedding status in columns
        # ====================================================================
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            # Display embedding status and metadata
            if embedding_exists:
                st.success("✅ Embeddings Available")
                st.caption(f"Project: {project_name}")
                st.caption(f"Files: {existing_metadata.get('file_count', 'Unknown')}")
                st.caption(f"Created: {existing_metadata.get('created_at', 'Unknown')}")
            else:
                st.warning("⚠️ No Embeddings Found")
                st.caption(f"Project: {project_name}")
                st.caption("Need to create embeddings first")
        
        with col2:
            # Button to create new embeddings (always available)
            if st.button("🔄 Create Embeddings", type="primary", use_container_width=True):
                st.session_state.create_embeddings = True  # Set flag for embedding creation
            
            # Button to delete existing embeddings (only if embeddings exist)
            if embedding_exists:
                if st.button("🗑️ Delete Embeddings", use_container_width=True):
                    if delete_embedding_data(project_name):
                        st.success("🗑️ Embeddings deleted!")
                        st.rerun()  # Refresh the page to update UI
        
        with col3:
            # Button to load existing embeddings (only if embeddings exist)
            if embedding_exists:
                if st.button("📊 Load Embeddings", type="secondary", use_container_width=True):
                    st.session_state.load_embeddings = True  # Set flag for loading embeddings
        
        # ====================================================================
        # EMBEDDING CREATION PROCESS: Create new embeddings when requested
        # ====================================================================
        if st.session_state.get('create_embeddings', False):
            with st.spinner("🔄 Creating embeddings for your codebase..."):
                # Clean up any existing Chroma instances to prevent conflicts
                cleanup_chroma_instances()
                
                # Initialize containers for documents and progress tracking
                documents = []
                file_count = 0
                
                # Create progress bar and status text for user feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each file extension
                for ext in file_extensions:
                    # Create search pattern for files with this extension
                    pattern = os.path.join(code_folder, f"**/*{ext}")
                    files = glob.glob(pattern, recursive=True)  # Find all matching files recursively
                    
                    # Process each file found
                    for i, file_path in enumerate(files):
                        try:
                            # Update status to show current file being processed
                            status_text.text(f"Loading: {os.path.basename(file_path)}")
                            
                            # Read file content with UTF-8 encoding, ignore invalid characters
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Skip empty files
                            if len(content.strip()) == 0:
                                continue
                                
                            # Create relative path for cleaner display
                            rel_path = os.path.relpath(file_path, code_folder)
                            
                            # Create a Document object with content and metadata
                            documents.append(Document(
                                page_content=f"// File: {rel_path}\n// Full path: {file_path}\n\n{content}",
                                metadata={
                                    'source': file_path,           # Full file path
                                    'relative_path': rel_path,     # Relative path for display
                                    'file_type': ext,              # File extension
                                    'file_name': os.path.basename(file_path),  # Just filename
                                    'file_size': len(content)      # Content size in characters
                                }
                            ))
                            file_count += 1
                            
                            # Update progress bar
                            progress_bar.progress(min((i + 1) / len(files), 1.0))
                            
                        except Exception as e:
                            # Log files that couldn't be read (permissions, encoding issues, etc.)
                            st.warning(f"Could not read {file_path}: {str(e)}")
                
                # Update status for text processing phase
                status_text.text("Creating text chunks...")
                
                if documents:
                    # ========================================================
                    # TEXT SPLITTING: Break large files into manageable chunks
                    # ========================================================
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=3000,    # Maximum characters per chunk
                        chunk_overlap=200,  # Overlap between chunks to maintain context
                        separators=[        # Split points optimized for C++ code
                            "\nclass ",     # Split at class definitions
                            "\nstruct ",    # Split at struct definitions
                            "\nnamespace ", # Split at namespace definitions
                            "\nvoid ",      # Split at function definitions
                            "\nint ",       # Split at function definitions
                            "\n\n",         # Split at double newlines
                            "\n"            # Split at single newlines if needed
                        ]
                    )
                    
                    # Split documents into chunks
                    split_docs = text_splitter.split_documents(documents)
                    
                    # Update status for embedding creation phase
                    status_text.text("Creating vector embeddings...")
                    
                    # ========================================================
                    # VECTOR STORE CREATION: Create embeddings and store them
                    # ========================================================
                    # Ensure cache directory exists
                    cache_dir = os.path.join(os.path.dirname(__file__), "embedding_cache")
                    
                    if not ensure_safe_directory(cache_dir):
                        st.error("❌ Cannot create or write to embedding cache directory. Check permissions.")
                        # Clean up UI elements and skip embedding creation
                        progress_bar.empty()
                        status_text.empty()
                        st.session_state.create_embeddings = False
                    else:
                        # Generate path for this project
                        vector_store_path = os.path.join(cache_dir, f"vectorstore_{project_name}")
                        
                        # Clean up any existing corrupted database files
                        if os.path.exists(vector_store_path):
                            cleanup_database_files(vector_store_path)
                        
                        # Ensure clean directory for new chromadb architecture
                        if os.path.exists(vector_store_path):
                            import shutil
                            shutil.rmtree(vector_store_path, ignore_errors=True)
                            import time
                            time.sleep(0.5)  # Wait for cleanup
                        
                        os.makedirs(vector_store_path, exist_ok=True)
                        
                        embedding_success = False
                        try:
                            # Clean up any existing Chroma instances to prevent conflicts
                            import gc
                            gc.collect()
                            
                            # Create FAISS vector database (no deprecated configuration issues)
                            from langchain_community.vectorstores import FAISS
                            import uuid
                            
                            # Create FAISS vector store from documents
                            vectorstore = FAISS.from_documents(
                                split_docs,
                                embeddings
                            )
                            
                            # Save FAISS index to disk
                            vectorstore.save_local(vector_store_path)
                            
                            # Test the database by performing a simple search
                            try:
                                test_results = vectorstore.similarity_search("test", k=1)
                                # If search works, the database is properly created
                                embedding_success = True
                                
                            except Exception as e:
                                st.error(f"❌ Database verification failed: {e}")
                                cleanup_database_files(vector_store_path)
                                embedding_success = False
                                
                        except Exception as e:
                            # If the error is about existing instances, try a more thorough cleanup
                            if "already exists" in str(e).lower():
                                st.warning("🔄 Existing database instance detected. Cleaning up...")
                                
                                # More aggressive cleanup
                                try:
                                    import gc
                                    
                                    # Force cleanup of the directory
                                    cleanup_database_files(vector_store_path)
                                    
                                    # Force garbage collection
                                    gc.collect()
                                    
                                    # Wait a moment for cleanup
                                    import time
                                    time.sleep(1)
                                    
                                    # Try creating the vectorstore again with FAISS
                                    from langchain_community.vectorstores import FAISS
                                    
                                    vectorstore = FAISS.from_documents(
                                        split_docs,
                                        embeddings
                                    )
                                    
                                    vectorstore.save_local(vector_store_path)
                                    
                                    # Test the database
                                    test_results = vectorstore.similarity_search("test", k=1)
                                    embedding_success = True
                                    st.success("🔄 Successfully recreated database after cleanup!")
                                    
                                except Exception as cleanup_error:
                                    st.error(f"❌ Failed to cleanup and recreate database: {cleanup_error}")
                                    cleanup_database_files(vector_store_path)
                                    embedding_success = False
                            else:
                                st.error(f"❌ Error creating vector database: {e}")
                                cleanup_database_files(vector_store_path)
                                embedding_success = False
                        
                        # ========================================================
                        # METADATA SAVING: Save information about this embedding (only if successful)
                        # ========================================================
                        if embedding_success:
                            metadata = {
                                'project_name': project_name,
                                'folder_path': code_folder,
                                'file_extensions': file_extensions,
                                'file_count': file_count,
                                'chunk_count': len(split_docs),
                                'vector_store_path': vector_store_path,
                                'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Save metadata to disk for future reference
                            save_embedding_metadata(project_name, metadata)
                            
                            # Clean up UI elements
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Embeddings created! Processed {file_count} files, {len(split_docs)} chunks")
                            
                            # ========================================================
                            # SESSION STATE SETUP: Prepare for chat functionality
                            # ========================================================
                            # Create retriever for similarity search (returns top 6 relevant chunks)
                            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                            st.session_state.embeddings_loaded = True  # Flag to enable chat interface
                            st.session_state.current_metadata = metadata
                        else:
                            # Clean up UI elements if embedding creation failed
                            progress_bar.empty()
                            status_text.empty()
                    
                else:
                    # Clean up UI if no files were processed
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ No valid code files found!")
            
            # Clear the create embeddings flag to prevent re-execution
            st.session_state.create_embeddings = False
        
        # ====================================================================
        # EMBEDDING LOADING PROCESS: Load previously created embeddings
        # ====================================================================
        elif st.session_state.get('load_embeddings', False):
            if embedding_exists:
                with st.spinner("📊 Loading existing embeddings..."):
                    # Clean up any existing Chroma instances to prevent conflicts
                    cleanup_chroma_instances()
                    
                    try:
                        # Get the path to the saved vector store
                        vector_store_path = existing_metadata['vector_store_path']
                        
                        # Verify the vector store directory exists and is accessible
                        if not os.path.exists(vector_store_path):
                            st.error(f"❌ Vector store directory not found: {vector_store_path}")
                            st.info("💡 Try recreating the embeddings.")
                        else:
# Load the existing FAISS vector database
                            try:
                                from langchain_community.vectorstores import FAISS
                                
                                # Load FAISS index from disk
                                vectorstore = FAISS.load_local(
                                    vector_store_path,
                                    embeddings,
                                    allow_dangerous_deserialization=True
                                )
                                
                                # Test the database by performing a simple search
                                test_results = vectorstore.similarity_search("test", k=1)
                                
                                # Setup retriever and session state for chat functionality
                                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                                st.session_state.embeddings_loaded = True
                                st.session_state.current_metadata = existing_metadata
                                
                                st.success(f"✅ Loaded embeddings for {existing_metadata['file_count']} files!")
                                
                            except Exception as db_error:
                                st.error(f"❌ Database corruption detected: {db_error}")
                                st.warning("🔧 Attempting to clean up corrupted database...")
                                
                                # Clean up corrupted database
                                cleanup_database_files(vector_store_path)
                                
                                # Also remove the metadata since the database is corrupted
                                cache_dir = os.path.join(os.path.dirname(__file__), "embedding_cache")
                                cache_file = os.path.join(cache_dir, f"embedding_meta_{project_name}.pkl")
                                if os.path.exists(cache_file):
                                    os.remove(cache_file)
                                
                                st.info("💡 Please recreate the embeddings.")
                            
                            st.info("💡 Please recreate the embeddings.")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading embeddings: {e}")
                        st.info("💡 Try recreating the embeddings if the problem persists.")
            
            # Clear the load embeddings flag
            st.session_state.load_embeddings = False
        
        # ====================================================================
        # CHAT INTERFACE: Only show if embeddings are loaded and ready
        # ====================================================================
        if st.session_state.get('embeddings_loaded', False):
            st.markdown("---")
            st.markdown("### 💬 Chat with your codebase:")
            
            # ================================================================
            # SESSION MANAGEMENT: Setup chat history storage with memory control
            # ================================================================
            if "store" not in st.session_state:
                st.session_state.store = {}  # Dictionary to store chat histories by session

            # Session ID input with history management
            col_session, col_clear = st.columns([3, 1])

            with col_session:
                session_id = st.text_input("Session ID:", value="cpp_session")

            with col_clear:
                st.write("")  # Add some spacing
                if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                    if session_id in st.session_state.store:
                        del st.session_state.store[session_id]
                        st.success("🧹 Chat history cleared!")
                        st.rerun()

            # Display current session history info
            if session_id in st.session_state.store:
                session_history = st.session_state.store[session_id]
                msg_count, size_bytes = get_history_size(session_history)
                
                if msg_count > 0:
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        st.metric("Messages", msg_count)
                    with col2:
                        st.metric("Memory", format_size(size_bytes))
                    with col3:
                        # Show warning if history gets large
                        if msg_count > 20:
                            st.warning("⚠️ Large History")
                        elif msg_count > 10:
                            st.info("📊 Medium History")
                        else:
                            st.success("✅ Small History")

            # Add history management options
            with st.expander("🧠 Memory Management Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Option to limit history size
                    max_history = st.slider(
                        "Max messages to keep in history:",
                        min_value=5,
                        max_value=50,
                        value=20,
                        help="Automatically trim old messages when limit is reached"
                    )
                    st.session_state.max_history = max_history
                
                with col2:
                    # Show all session statistics
                    if st.session_state.store:
                        st.write("**All Sessions:**")
                        total_messages = 0
                        total_size = 0
                        for sid, hist in st.session_state.store.items():
                            count, size = get_history_size(hist)
                            total_messages += count
                            total_size += size
                            st.write(f"• {sid}: {count} msgs ({format_size(size)})")
                        
                        st.write(f"**Total: {total_messages} messages ({format_size(total_size)})**")
                        
                        if st.button("🗑️ Clear All Sessions", type="secondary"):
                            st.session_state.store = {}
                            st.success("🧹 All chat history cleared!")
                            st.rerun()
            
            # ================================================================
            # STATISTICS DISPLAY: Show information about loaded codebase
            # ================================================================
            if hasattr(st.session_state, 'current_metadata'):
                metadata = st.session_state.current_metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files", metadata.get('file_count', 'N/A'))
                with col2:
                    st.metric("Chunks", metadata.get('chunk_count', 'N/A'))
                with col3:
                    st.metric("Extensions", str(len(metadata.get('file_extensions', []))))
            
            # ================================================================
            # RAG CHAIN SETUP: Create the conversation chain with history
            # ================================================================
            
            # STEP 1: Create prompt for contextualizing questions based on chat history
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question about C++ code, "
                "reformulate the question to be standalone if needed."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
                ("human", "{input}")                                # Placeholder for user input
            ])

            # STEP 2: Create history-aware retriever that considers chat context
            # This retriever will reformulate questions based on conversation history
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,                           # Language model for question reformulation
                retriever=st.session_state.retriever,  # Vector database retriever
                prompt=contextualize_q_prompt      # Prompt for contextualizing questions
            )

            # STEP 3: Create prompt template for final answer generation
            system_prompt = (
                "You are a C++ Code Expert analyzing the user's codebase. "
                "Based on the provided code context, give detailed, actionable advice.\n\n"
                "Code Context:\n{context}"        # Placeholder for retrieved documents
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),  # Chat history context
                ("human", "{input}")                   # User's question
            ])

            # STEP 4: Create document processing chain
            # This combines retrieved documents with the prompt template
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
            
            # STEP 5: Create the complete RAG chain
            # This orchestrates the entire retrieval-augmented generation process
            rag_chain = create_retrieval_chain(
                retriever=history_aware_retriever,  # Context-aware document retriever
                combine_docs_chain=document_chain   # Document combination and answer generation
            )

            # STEP 6: Define chat history management function with automatic trimming
            def get_session_chat_history(session_id: str) -> BaseChatMessageHistory:
                """
                Get or create chat history for a specific session with automatic trimming.
                This allows multiple concurrent conversations while managing memory.
                """
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                
                history = st.session_state.store[session_id]
                
                # Trim history if it exceeds the maximum size
                max_history = getattr(st.session_state, 'max_history', 20)
                if len(history.messages) > max_history:
                    # Keep only the most recent messages
                    messages_to_keep = history.messages[-max_history:]
                    history.clear()
                    for msg in messages_to_keep:
                        history.add_message(msg)
                
                return history

            # STEP 7: Create conversation chain with message history
            # This wraps the RAG chain to automatically manage chat history
            conversation_rag_history = RunnableWithMessageHistory(
                rag_chain,                         # The RAG chain to wrap
                get_session_chat_history,          # Function to get chat history
                input_messages_key="input",        # Key for user input
                history_messages_key="chat_history",  # Key for chat history
                output_messages_key="answer"       # Key for assistant's response
            )
            
            # ================================================================
            # QUICK QUESTION BUTTONS: Pre-defined questions for common tasks
            # ================================================================
            st.markdown("**💡 Quick Questions:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Code Overview", use_container_width=True):
                    st.session_state.suggested_question = "Give me an overview of this codebase structure and main components."
            with col2:
                if st.button("🔧 Find Issues", use_container_width=True):
                    st.session_state.suggested_question = "Analyze this code for potential bugs, improvements, and best practices."
            with col3:
                if st.button("🚀 Add Feature", use_container_width=True):
                    st.session_state.suggested_question = "How can I add a new feature following the existing code patterns?"
            
            # ================================================================
            # USER QUESTION INPUT: Text area for user to ask questions
            # ================================================================
            user_question = st.text_area(
                "Ask about your codebase:",
                value=getattr(st.session_state, 'suggested_question', ''),  # Pre-fill if suggested question was clicked
                height=100,
                placeholder="Ask about specific functions, classes, or request improvements..."
            )
            
            # ================================================================
            # QUESTION PROCESSING: Handle user questions and generate responses
            # ================================================================
            if st.button("🤖 Ask C++ Expert", type="primary") and user_question:
                with st.spinner("Analyzing..."):
                    try:
                        # Invoke the conversation chain with the user's question
                        response = conversation_rag_history.invoke(
                            {"input": user_question},                          # User's question
                            config={"configurable": {"session_id": session_id}}  # Session configuration
                        )
                        
                        # Display the expert's response
                        st.markdown("### 🎯 Expert Analysis:")
                        st.markdown(response['answer'])
                        
                        # ====================================================
                        # REFERENCE FILES DISPLAY: Show which files were used
                        # ====================================================
                        if 'context' in response and response['context']:
                            with st.expander("📁 Referenced files:"):
                                referenced_files = set()
                                # Extract unique file paths from the retrieved documents
                                for doc in response['context']:
                                    if 'relative_path' in doc.metadata:
                                        referenced_files.add(doc.metadata['relative_path'])
                                
                                # Display the referenced files in sorted order
                                for file_path in sorted(referenced_files):
                                    st.write(f"• `{file_path}`")
                        
                        # Clear any suggested question after processing
                        if 'suggested_question' in st.session_state:
                            del st.session_state.suggested_question
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        else:
            # ================================================================
            # INSTRUCTION MESSAGE: Show when embeddings need to be created/loaded
            # ================================================================
            st.info("👆 Create or load embeddings first to start chatting with your codebase!")
    
    # ========================================================================
    # ERROR HANDLING: Handle missing inputs or invalid paths
    # ========================================================================
    elif not project_name:
        st.info("👆 Enter a project name to get started")
    elif code_folder and not os.path.exists(code_folder):
        st.error("❌ Folder path does not exist.")
    else:
        st.info("👆 Enter your project name, C++ code folder path, and select file types to get started")

# ============================================================================
# FOOTER: Privacy and local processing notice
# ============================================================================
st.markdown("---")
st.markdown("*🔒 All processing happens locally on your machine*")