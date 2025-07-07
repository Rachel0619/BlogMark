import streamlit as st
import os
from search import BlogRAGHandler

# Initialize session state
if 'rag_handler' not in st.session_state:
    st.session_state.rag_handler = None

@st.cache_resource
def get_rag_handler():
    """Initialize and cache the RAG handler"""
    try:
        # Initialize with empty documents to avoid loading during init
        handler = BlogRAGHandler.__new__(BlogRAGHandler)
        
        # Manually initialize the necessary components
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
        
        handler.urls_file = 'urls.csv'
        handler.collection_name = os.environ.get('COLLECTION_NAME')
        handler.top_k = 3
        
        # Initialize Qdrant client
        import qdrant_client
        handler.qdrant_client = qdrant_client.QdrantClient(url=os.environ.get('QDRANT_URL'))
        
        # Set up vector store
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        handler.qdrant_vector_store = QdrantVectorStore(
            collection_name=handler.collection_name,
            client=handler.qdrant_client,
            fastembed_sparse_model=os.environ.get("sparse_model"),
            enable_hybrid=True
        )
        
        # Load existing index
        from llama_index.core import VectorStoreIndex
        handler.vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store=handler.qdrant_vector_store,
            embed_model=OllamaEmbedding(model_name=os.environ.get("EMBED_MODEL_NAME"))
        )
        
        return handler
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def initialize_rag_handler():
    """Initialize the RAG handler if not already done"""
    if st.session_state.rag_handler is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_handler = get_rag_handler()
            if st.session_state.rag_handler:
                st.success("RAG system initialized successfully!")
                return True
            else:
                return False
    return True

def main():
    st.set_page_config(
        page_title="BlogMark - AI-Powered Blog Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 BlogMark - AI-Powered Blog Search")
    st.markdown("Search through your blog collection using natural language queries")
    
    # Initialize RAG handler
    if not initialize_rag_handler():
        st.stop()
    
    # Main query interface
    st.header("🔍 Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is tiny agent?",
        help="Ask any question about the content in your blog collection"
    )
    
    # Search button
    if st.button("Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching through your blog collection..."):
                try:
                    # Perform the search
                    response = st.session_state.rag_handler.query(query)
                    
                    # Display results
                    st.header("📝 Answer")
                    st.markdown(response.response)
                    
                    # Show source information if available
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        st.header("📚 Sources")
                        for i, node in enumerate(response.source_nodes[:3]):  # Show top 3 sources
                            with st.expander(f"Source {i+1}: {node.metadata.get('title', 'Unknown Title')}"):
                                st.write(f"**URL:** {node.metadata.get('url', 'Unknown')}")
                                st.write(f"**Author:** {node.metadata.get('author', 'Unknown')}")
                                st.write(f"**Date:** {node.metadata.get('date', 'Unknown')}")
                                st.write(f"**Site:** {node.metadata.get('sitename', 'Unknown')}")
                                st.write("**Content Preview:**")
                                st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                                
                except Exception as e:
                    st.error(f"An error occurred while searching: {str(e)}")
        else:
            st.warning("Please enter a question to search.")
    
    # Future features section (placeholder)
    st.markdown("---")
    st.header("🚀 Coming Soon")
    st.info("**Add New Blog URLs** - Soon you'll be able to add new blog URLs directly from this interface and they'll be automatically processed and added to your search collection.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ Information")
        st.write("This AI-powered search system helps you find relevant information from your blog collection.")
        
        st.subheader("🔧 System Status")
        if st.session_state.rag_handler:
            st.success("✅ RAG System: Online")
            st.success("✅ Vector Database: Connected")
            st.success("✅ LLM: Ready")
        else:
            st.error("❌ System not initialized")
            
        st.subheader("📊 Statistics")
        if st.session_state.rag_handler:
            try:
                # This is a placeholder - you might want to add actual statistics
                st.metric("Collection", "blog_posts")
                st.metric("Search Type", "Hybrid (Dense + Sparse)")
                st.metric("Top-K Results", "3")
            except:
                st.write("Statistics unavailable")

if __name__ == "__main__":
    main()