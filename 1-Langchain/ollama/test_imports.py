import sys
import os

# Add the directory to sys.path to test imports
sys.path.append(r'c:\UdemyAICourse\Langchain\1-Langchain\ollama')

try:
    print("Testing imports...")
    
    # Test basic streamlit import
    import streamlit as st
    print("✅ Streamlit imported successfully")
    
    # Test langchain imports
    from langchain_ollama import OllamaLLM
    print("✅ LangChain Ollama imported successfully")
    
    from langchain_huggingface import HuggingFaceEmbeddings  
    print("✅ HuggingFace embeddings imported successfully")
    
    from langchain_community.vectorstores import FAISS
    print("✅ FAISS imported successfully")
    
    # Test that the main script compiles
    with open(r'c:\UdemyAICourse\Langchain\1-Langchain\ollama\coderagent.py', 'r') as f:
        code = f.read()
    
    compile(code, 'coderagent.py', 'exec')
    print("✅ Code compiles without syntax errors")
    
    print("\n🎉 All basic tests passed!")
    print("The app should now run without crashing on startup.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()