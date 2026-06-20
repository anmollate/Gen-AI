import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)


st.set_page_config(
    page_title="Standard RAG System",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# One-time setup (cached)
@st.cache_resource
def load_nltk():
    nltk.download("punkt_tab", quiet=True)
    return True

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

@st.cache_resource
def load_chat_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

load_nltk()

# Title
st.title("🔍 Standard RAG System")
st.caption("Retrieval-Augmented Generation pipeline demo")

st.divider()


# Architecture + Model Info Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    **Pipeline Flow:**

    1. **Sentence Tokenization** — Input text is split into sentences (`nltk.sent_tokenize`) and cleaned.
    2. **Embedding Generation** — Each sentence is embedded individually using `BAAI/bge-m3` (batched via `embed_documents`).
    3. **Semantic Chunking** — Cosine similarity is computed between consecutive sentence embeddings. A chunk boundary is placed wherever similarity drops below `mean − std_dev` of all similarity scores — i.e., chunks break at topic shifts, not at fixed lengths.
    4. **Vector Storage** — Each chunk is re-embedded and stored in **ChromaDB** with an auto-incrementing ID.
    5. **Query Rewriting** — The user's raw query is rewritten by an LLM into a search-optimized form (intent-preserving, no outside info added).
    6. **Retrieval** — The rewritten query is embedded and matched against ChromaDB via cosine similarity (top-3 chunks).
    7. **Augmented Generation** — Retrieved chunks are passed as strict context to the LLM, which must answer **only** from that context (or say it doesn't know).
    """)

with col2:
    st.subheader("🧠 Models Used")
    st.markdown("""
    **Embedding Model**
    - `BAAI/bge-m3` via `HuggingFaceEndpointEmbeddings`
    - Used for sentence embeddings, chunk embeddings, and query embeddings

    **LLM**
    - `meta-llama/Llama-3.1-8B-Instruct` via `HuggingFaceEndpoint` + `ChatHuggingFace`
    - Used twice: once for **query rewriting**, once for **grounded answer generation**

    **Vector Store**
    - ChromaDB (in-memory `chromadb.Client()`), one collection per session
    """)

    st.info("**Standard RAG + Query Rewriting.** Chunking is semantic (similarity-drop based), not fixed-size.", icon="ℹ️")

st.divider()


# Pipeline Functions (from notebook, with chunking fixes)

def tokenize_and_clean(text: str) -> list[str]:
    sentences = sent_tokenize(text)
    cleaned = [s.replace("\n", " ") for s in sentences]
    return cleaned


def compute_similarity_scores(embeddings: list) -> list[tuple[int, int, float]]:
    """Cosine similarity between every pair of consecutive sentence embeddings.
    Fix: original notebook loop stopped one pair short (range went to len-1
    inclusive only by chance of the while condition) — using range() here
    guarantees every consecutive pair (0,1), (1,2), ..., (n-2, n-1) is covered.
    """
    scores = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        scores.append((i, i + 1, sim))
    return scores


def semantic_chunk_indices(similarity_scores: list[tuple[int, int, float]]) -> list[list[int]]:
    """Group sentence indices into chunks, breaking where similarity drops
    below mean - std. Fix: guard against a degenerate threshold (e.g. when
    std is 0, or text homogeneity collapses everything into one chunk, or
    every sentence becomes its own chunk) by clamping the threshold to a
    sane percentile-based fallback.
    """
    sims = [s[2] for s in similarity_scores]
    mean = np.mean(sims)
    std = np.std(sims)
    threshold = mean - std

    # Guard: if threshold is degenerate (too low -> 1 giant chunk,
    # too high -> every sentence is its own chunk), fall back to the
    # 25th percentile of similarity scores as a safer split point.
    p25 = np.percentile(sims, 25)
    if threshold <= 0 or threshold >= max(sims, default=1):
        threshold = p25

    chunks = []
    current_chunk = [0]  # first chunk always starts at sentence 0

    for (i, j, sim) in similarity_scores:
        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(j)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_chunk_texts(cleaned_sentences: list[str], chunk_indices: list[list[int]]) -> list[str]:
    sentence_lookup = {idx: sent for idx, sent in enumerate(cleaned_sentences)}
    chunk_texts = []
    for chunk in chunk_indices:
        chunk_sentences = [sentence_lookup[idx] for idx in chunk if idx in sentence_lookup]
        if chunk_sentences:
            chunk_texts.append(" ".join(chunk_sentences))
    return chunk_texts


def rewrite_query(chat_model, raw_query: str) -> str:
    prompt = f"""
You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite the user's query so that it is clear, specific, and optimized for semantic search in a vector database.

Guidelines:
1. Preserve the original intent of the user.
2. Do not answer the question.
3. Do not introduce information not implied by the query.
4. Return only the rewritten query and nothing else.
5. Do Not Add Any Extra Information To The Query From Your End Just Improve The Semantics Of The Query
6. You Know Nothing About The Context Of The Query

User Query:
{raw_query}

Rewritten Query:
"""
    result = chat_model.invoke(prompt)
    return result.content.strip()


def generate_answer(chat_model, context_docs: list, question: str) -> str:
    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.
Provide As Much Context As Possible While Answering The Query
If the answer is not present in the context, say:
"I don't know based on the provided context."

Context:
{context_docs}

Question:
{question}

Answer:
"""
    result = chat_model.invoke(prompt)
    return result.content


def run_rag_pipeline(raw_text: str, raw_query: str, status_callback=None):
    """Full pipeline: chunk -> embed -> store -> rewrite query -> retrieve -> generate."""

    def report(msg):
        if status_callback:
            status_callback(msg)

    embedding_model = load_embedding_model()
    chat_model = load_chat_model()

    # 1. Tokenize + clean
    report("Tokenizing and cleaning sentences...")
    cleaned_sentences = tokenize_and_clean(raw_text)

    if len(cleaned_sentences) < 2:
        return "⚠️ Need at least 2 sentences in the input text to perform semantic chunking.", None

    # 2. Embed sentences
    report("Embedding sentences with bge-m3...")
    sentence_embeddings = embedding_model.embed_documents(cleaned_sentences)

    # 3. Similarity scores between consecutive sentences
    report("Computing inter-sentence similarity...")
    similarity_scores = compute_similarity_scores(sentence_embeddings)

    # 4. Semantic chunking
    report("Performing semantic chunking...")
    chunk_indices = semantic_chunk_indices(similarity_scores)
    chunk_texts = build_chunk_texts(cleaned_sentences, chunk_indices)

    if not chunk_texts:
        return "⚠️ Chunking produced no valid chunks. Try providing more text.", None

    # 5. Store in ChromaDB (fresh in-memory collection per run)
    report("Storing chunks in ChromaDB...")
    client = chromadb.Client()
    collection_name = f"rag_session_{abs(hash(raw_text)) % (10 ** 8)}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    chunk_embeddings = embedding_model.embed_documents(chunk_texts)
    collection.add(
        ids=[str(i) for i in range(len(chunk_texts))],
        documents=chunk_texts,
        embeddings=chunk_embeddings
    )

    # 6. Query rewriting
    report("Rewriting query for better semantic search...")
    enhanced_query = rewrite_query(chat_model, raw_query)

    # 7. Retrieval
    report("Retrieving relevant chunks...")
    query_embedding = embedding_model.embed_query(enhanced_query)
    n_results = min(3, len(chunk_texts))
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    retrieved_docs = result["documents"]

    # 8. Generation
    report("Generating grounded answer...")
    answer = generate_answer(chat_model, retrieved_docs, enhanced_query)

    debug_info = {
        "num_sentences": len(cleaned_sentences),
        "num_chunks": len(chunk_texts),
        "chunks": chunk_texts,
        "enhanced_query": enhanced_query,
        "retrieved_docs": retrieved_docs,
    }

    return answer, debug_info


# Interactive Demo Section

st.subheader("⚙️ Try It Out")

if not HUGGINGFACEHUB_API_TOKEN:
    st.warning(
        "⚠️ `HUGGINGFACEHUB_API_TOKEN` not found. Create a `.env` file next to this script with:\n\n"
        "`HUGGINGFACEHUB_API_TOKEN=your_token_here`",
        icon="🔑"
    )

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Step 1: Provide Knowledge Source**")
    user_text = st.text_area(
        "Paste the text you want the RAG system to learn from:",
        height=220,
        placeholder="Paste an article, document, or any reference text here..."
    )

    st.markdown("**Step 2: Ask a Question**")
    user_query = st.text_input(
        "Enter your query:"
    )

    show_debug = st.checkbox("Show pipeline details (chunks, rewritten query, retrieved context)")

    run_button = st.button("Run RAG Pipeline 🚀", type="primary", use_container_width=True)

with right_col:
    st.markdown("**Output: LLM Answer**")
    output_box = st.empty()
    status_box = st.empty()
    debug_box = st.container()

    if run_button:
        if not HUGGINGFACEHUB_API_TOKEN:
            output_box.error("❌ Cannot run pipeline without a valid HuggingFace API token.")
        elif not user_text.strip():
            output_box.warning("⚠️ Please provide some text as the knowledge source.")
        elif not user_query.strip():
            output_box.warning("⚠️ Please enter a query.")
        else:
            try:
                with st.spinner("Running pipeline..."):
                    answer, debug_info = run_rag_pipeline(
                        user_text,
                        user_query,
                        status_callback=lambda msg: status_box.caption(f"⏳ {msg}")
                    )
                status_box.empty()
                output_box.success(answer)

                if show_debug and debug_info:
                    with debug_box:
                        st.markdown("---")
                        st.markdown(f"**Sentences:** {debug_info['num_sentences']} → **Chunks:** {debug_info['num_chunks']}")
                        with st.expander("📦 Semantic Chunks"):
                            for i, c in enumerate(debug_info["chunks"]):
                                st.markdown(f"**Chunk {i}:** {c}")
                        with st.expander("✏️ Rewritten Query"):
                            st.write(debug_info["enhanced_query"])
                        with st.expander("📥 Retrieved Context"):
                            st.write(debug_info["retrieved_docs"])
            except Exception as e:
                status_box.empty()
                output_box.error(f"❌ Pipeline error: {e}")
    else:
        output_box.markdown("*Your answer will appear here after running the pipeline.*")

st.divider()
st.caption("Standard RAG · Semantic Chunking + bge-m3 + ChromaDB + Llama-3.1-8B-Instruct (query rewrite + generation)")