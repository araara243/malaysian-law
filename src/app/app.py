"""
Malaysian Legal RAG - Streamlit Web Interface

A user-friendly interface for querying Malaysian legal acts
with AI-powered answers and source citations.

Features:
- Chat-based Q&A interface
- Source citations with expandable sections
- Support for Contracts Act, Specific Relief Act, Housing Development Act
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from generation.rag_chain import LegalRAGChain
from retrieval.hybrid_retriever import HybridRetriever


# Page configuration
st.set_page_config(
    page_title="Malaysian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .source-title {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .stChatMessage {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_chain():
    """Load and cache the RAG chain."""
    return LegalRAGChain(
        model_name="openrouter/free",  # Auto-routes to best available free model
        temperature=0.1,
        n_results=5,
        retrieval_method="hybrid",
        llm_provider="openrouter"
    )


def render_sidebar():
    """Render the sidebar with info and settings."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Coat_of_arms_of_Malaysia.svg/200px-Coat_of_arms_of_Malaysia.svg.png", width=100)
        
        st.markdown("## ⚖️ Malaysian Legal Assistant")
        
        st.markdown("""
        ### 📚 Available Acts
        **Commercial Law:**
        - Contracts Act 1950 (Act 136)
        - Specific Relief Act 1951 (Act 137)
        - Partnership Act 1961 (Act 135)
        - Sale of Goods Act 1957 (Act 383)

        **Property Law:**
        - Housing Development (Control and Licensing) Act 1966 (Act 118)
        - Strata Titles Act 1985 (Act 318)

        **Civil Procedure:**
        - Courts of Judicature Act 1964 (Act 91)
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 💡 Example Questions

        **Contract Law:**
        - What is the definition of consideration in contract law?
        - When is a contract voidable due to coercion?
        - What constitutes free consent in contracts?
        - What is fraud under the Contracts Act?

        **Property Law:**
        - What are the licensing requirements for housing developers?
        - What is a Housing Development Account?
        - What powers does the Controller have over developers?

        **General:**
        - When can specific performance be ordered by a court?
        - What remedies are available for breach of contract?
        """)
        
        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("Show source sections", value=True)
        show_stats = st.checkbox("Enable response logging", value=False,
                                        help="Log responses for model comparison and statistics")

        st.markdown("""
        ### 🤖 AI Model
        **Provider:** OpenRouter (Free Tier)
        **Model:** Auto-routing to best free model

        Get your free API key at [openrouter.ai](https://openrouter.ai/keys)
        """)

        st.markdown("---")
        
        st.markdown("""
        ### ⚠️ Disclaimer
        This tool provides general legal information only. 
        It is **not** a substitute for professional legal advice.
        Always consult a qualified lawyer for specific legal matters.
        """)
        
        return show_sources


def render_sources(sources: list):
    """Render source citations in an expandable format."""
    if not sources:
        return
    
    st.markdown("### 📖 Sources")
    
    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            act_name = source.get("act_name", "Unknown")
            section = source.get("section_number", "?")
            title = source.get("section_title", "")
            score = source.get("score", 0)
        else:
            act_name = source.act_name
            section = source.section_number
            title = source.section_title
            score = source.score
        
        with st.expander(f"📄 Source {i}: {act_name}, Section {section}", expanded=False):
            st.markdown(f"**Act:** {act_name}")
            st.markdown(f"**Section:** {section}")
            if title:
                st.markdown(f"**Title:** {title}")
            st.markdown(f"**Relevance Score:** {score:.4f}")
            
            # Show content if available
            if hasattr(source, 'content'):
                st.markdown("---")
                st.markdown("**Full Text:**")
                st.text(source.content[:500] + "..." if len(source.content) > 500 else source.content)


def main():
    """Main application."""
    # Render sidebar
    show_sources, show_stats = render_sidebar()

    # Main content
    st.markdown('<p class="main-header">⚖️ Malaysian Legal Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about Malaysian Contracts, Specific Relief, and Housing Development law</p>', unsafe_allow_html=True)

    # Model info (transparency)
    from generation.rag_chain import LegalRAGChain
    model_info = LegalRAGChain.get_model_info()

    # Display model info
    st.caption(
        f"🤖 AI Provider: {model_info['provider'].upper()} | "
        f"Model: {model_info['model']} | "
        f"Temp: {model_info['temperature']} | "
        f"{model_info['description']}"
    )
    st.markdown("---")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []
    
    # Load RAG chain
    try:
        rag_chain = load_rag_chain()
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        st.stop()
    
    # Create layout with chat and sources
    if show_sources and st.session_state.sources:
        col1, col2 = st.columns([2, 1])
    else:
        col1 = st.container()
        col2 = None
    
    with col1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a legal question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Researching Malaysian law..."):
                    try:
                        result = rag_chain.ask(prompt, return_sources=True, log_response=show_stats)
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(answer)
                        
                        # Store sources for display
                        st.session_state.sources = sources
                        
                    except Exception as e:
                        error_str = str(e)
                        # Handle any API errors gracefully - fall back to retrieval only
                        st.warning(f"⚠️ LLM unavailable ({error_str[:100]}...). Showing retrieved sections:")
                        # Fall back to retrieval only
                        sources = rag_chain.retrieve(prompt)
                        context = rag_chain._retriever.format_context(sources)
                        answer = f"**Retrieved Legal Sections:**\n\n{context}"
                        st.markdown(answer)
                        st.session_state.sources = sources
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Rerun to update layout
            st.rerun()
    
    # Show sources in sidebar column
    if col2 and show_sources and st.session_state.sources:
        with col2:
            render_sources(st.session_state.sources)
    
    # Footer disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Legal Disclaimer:</strong> This AI assistant provides general information based on 
        Malaysian statutory law. It does not constitute legal advice. For specific legal matters, 
        please consult a qualified Malaysian lawyer.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
