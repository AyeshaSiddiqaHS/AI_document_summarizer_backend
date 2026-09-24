import os
import pdfplumber
import docx
import torch
import torch.nn.functional as F
from huggingface_hub import InferenceClient
from app.core.database import get_database
from app.core.config import settings

# Global variables for caching models
_device = None
_embedding_model = None
_gen_tokenizer = None
_gen_model = None

def _load_models():
    global _embedding_model, _gen_tokenizer, _gen_model, _device
    if _embedding_model is None or _gen_model is None:
        print("Loading AI Models...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from sentence_transformers import SentenceTransformer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
        hf_token = settings.HF_TOKEN if settings.HF_TOKEN else None
        
        print("Loading Embedding Model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=_device, token=hf_token)
        
        import transformers
        transformers.logging.set_verbosity_error()
        
        print("Loading Generation Model: google/flan-t5-base")
        _gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", token=hf_token)
        _gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", token=hf_token)
        _gen_model.to(_device)
        
        print(f"Models loaded successfully on {_device}")

def extract_text_from_file(file_path: str, file_format: str) -> str:
    """Extracts raw text from PDF, DOCX, or TXT."""
    text = ""
    file_format = file_format.upper()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_format == "PDF":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file_format in ["DOCX", "DOC"]:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_format == "TXT":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                text = f.read()
    else:
        raise ValueError("Unsupported file format.")
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Splits text into overlapping chunks of words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

async def generate_summary(doc_id: str, file_path: str, file_format: str, length: str = "medium") -> str:
    """Generates a high-quality summary using Qwen 2.5 / Llama 3.3 with local fallback.
    Supported length options: 'short', 'medium', 'long'.
    """
    length = (length or "medium").lower()
    if length not in ["short", "medium", "long"]:
        length = "medium"

    try:
        text = extract_text_from_file(file_path, file_format)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "Failed to read document."
        
    if not text:
        return "No text available to summarize in this document."

    hf_token = settings.HF_TOKEN if settings.HF_TOKEN else None

    # Strict Length-specific configurations for exact word count targets
    if length == "short":
        length_desc = (
            "Provide a SHORT summary. Length requirement: STRICTLY approximately 120 words (3 to 4 clear bullet points). "
            "Focus strictly on delivering a concise, punchy 120-word executive summary of the document's core purpose, findings, and conclusion."
        )
        max_tokens_budget = 240
    elif length == "long":
        length_desc = (
            "Provide a LONG, COMPREHENSIVE, AND IN-DEPTH summary. Length requirement: STRICTLY approximately 600 words. "
            "Organize thoroughly into clear titled sections (e.g. Executive Summary, Main Topics & Analysis, Key Findings & Data, Implications & Recommendations). "
            "Elaborate thoroughly with arguments, metrics, and details so the summary reaches approximately 600 words in total."
        )
        max_tokens_budget = 1100
    else:  # medium
        length_desc = (
            "Provide a MEDIUM summary. Length requirement: STRICTLY approximately 250 words. "
            "Provide a balanced, structured summary with a clear introductory context, bullet points for key takeaways, and a conclusion reaching approximately 250 words in total."
        )
        max_tokens_budget = 550

    # 1. Attempt summarization using cloud LLM (Qwen 2.5 / Llama 3.3) for fast, fluent, coherent summaries
    if hf_token:
        try:
            client = InferenceClient(api_key=hf_token)
            # Take up to first 4,000 words to respect prompt context window
            words = text.split()
            truncated_text = " ".join(words[:4500])
            
            prompt = (
                f"{length_desc}\n\n"
                f"Document Content:\n{truncated_text}"
            )
            
            # Try Qwen 2.5 72B first, fallback to Llama 3.3 70B
            models_to_try = ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct"]
            for model_name in models_to_try:
                try:
                    response = client.chat_completion(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert document summarizer. Create clear, beautifully formatted summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens_budget,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()
                    if summary:
                        return summary
                except Exception as m_err:
                    print(f"Model {model_name} failed: {m_err}")
                    continue
        except Exception as cloud_err:
            print(f"HuggingFace API summarization error, trying local fallback: {cloud_err}")
        
    # 2. Fallback to local model
    try:
        db = get_database()
        _load_models()
        
        cached_doc = None
        if db is not None:
            try:
                cached_doc = await db["document_chunks"].find_one({"doc_id": str(doc_id)})
            except Exception as dbe:
                print(f"DB access error: {dbe}")
        
        if cached_doc and "chunks" in cached_doc and "embeddings" in cached_doc:
            chunks = cached_doc["chunks"]
        else:
            chunks = chunk_text(text, chunk_size=300, overlap=50)
            chunk_embeddings = _embedding_model.encode(chunks, convert_to_tensor=True)
            
            if db is not None:
                try:
                    await db["document_chunks"].update_one(
                        {"doc_id": str(doc_id)},
                        {"$set": {
                            "chunks": chunks,
                            "embeddings": chunk_embeddings.tolist()
                        }},
                        upsert=True
                    )
                except Exception as dbe:
                    print(f"DB cache save error: {dbe}")
        
        chunks_to_summarize = chunks
        if len(chunks_to_summarize) > 8:
            if length == "short":
                chunks_to_summarize = chunks_to_summarize[:3] + chunks_to_summarize[-1:]
            else:
                chunks_to_summarize = chunks_to_summarize[:6] + chunks_to_summarize[-2:]
            
        chunk_summaries = []
        for chunk in chunks_to_summarize:
            prompt = f"Summarize the following text:\n\n{chunk}"
            inputs = _gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(_device)
            outputs = _gen_model.generate(**inputs, max_length=150, min_length=30)
            summary = _gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            chunk_summaries.append(summary)
            
        final_text = " ".join(chunk_summaries)
        
        if len(chunk_summaries) > 1:
            meta_prompt = f"Write a {length} summary of the following document summaries:\n\n{final_text}"
            inputs = _gen_tokenizer(meta_prompt, return_tensors="pt", max_length=1024, truncation=True).to(_device)
            target_max = 150 if length == "short" else (350 if length == "long" else 250)
            outputs = _gen_model.generate(**inputs, max_length=target_max, min_length=40)
            final_summary = _gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return final_summary
        else:
            return final_text

    except Exception as e:
        print(f"Error during local summarization: {e}")
        return "An error occurred while generating the summary."

async def generate_key_insights(doc_id: str, file_path: str, file_format: str) -> str:
    """Generates structured key insights, core metrics, and strategic takeaways."""
    try:
        text = extract_text_from_file(file_path, file_format)
    except Exception as e:
        print(f"Error extracting text for insights: {e}")
        return "Failed to read document."

    if not text:
        return "No text available to extract key insights."

    hf_token = settings.HF_TOKEN if settings.HF_TOKEN else None

    if hf_token:
        try:
            client = InferenceClient(api_key=hf_token)
            words = text.split()
            truncated_text = " ".join(words[:4000])

            prompt = (
                "Analyze the following document and extract the top critical Key Insights and Actionable Takeaways. "
                "Format clearly using categories like: \n"
                "- 💡 Core Findings & Key Insights\n"
                "- 📊 Essential Metrics, Data & Highlights\n"
                "- 🎯 Strategic Recommendations & Next Steps\n\n"
                "Keep the points concise, impactful, and clearly structured with emojis and bullet points.\n\n"
                f"Document Content:\n{truncated_text}"
            )

            models_to_try = ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct"]
            for model_name in models_to_try:
                try:
                    response = client.chat_completion(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a senior data analyst and strategic consultant. Extract powerful, high-value key insights."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=600,
                        temperature=0.3
                    )
                    insights = response.choices[0].message.content.strip()
                    if insights:
                        return insights
                except Exception as m_err:
                    print(f"Key insights model {model_name} failed: {m_err}")
                    continue
        except Exception as cloud_err:
            print(f"HuggingFace API key insights error: {cloud_err}")

    # Fallback to local prompt
    return await ask_document_question(
        doc_id=doc_id, 
        file_path=file_path, 
        file_format=file_format, 
        question="What are the top 5 most important key insights, metrics, and takeaways from this document?"
    )

async def ask_document_question(doc_id: str, file_path: str, file_format: str, question: str) -> str:
    """Answers a question using RAG: Embed chunks -> Search -> Qwen 2.5 / Llama 3.3 (with local FLAN-T5 fallback)."""
    if not question.strip():
        return "Please ask a valid question."
        
    db = get_database()
    
    try:
        _load_models()
        
        # Check cache in DB if db is available
        cached_doc = None
        if db is not None:
            try:
                cached_doc = await db["document_chunks"].find_one({"doc_id": str(doc_id)})
            except Exception as dbe:
                print(f"DB access error: {dbe}")
        
        if cached_doc and "chunks" in cached_doc and "embeddings" in cached_doc:
            # Cache Hit
            chunks = cached_doc["chunks"]
            chunk_embeddings = torch.tensor(cached_doc["embeddings"], device=_device)
        else:
            # Cache Miss
            try:
                text = extract_text_from_file(file_path, file_format)
            except Exception as e:
                print(f"Error extracting text: {e}")
                return "Failed to read document."
                
            if not text:
                return "I could not find this information in the uploaded document."
                
            chunks = chunk_text(text, chunk_size=300, overlap=50)
            chunk_embeddings = _embedding_model.encode(chunks, convert_to_tensor=True)
            
            # Save to DB if connected
            if db is not None:
                try:
                    await db["document_chunks"].update_one(
                        {"doc_id": str(doc_id)},
                        {"$set": {
                            "chunks": chunks,
                            "embeddings": chunk_embeddings.tolist()
                        }},
                        upsert=True
                    )
                except Exception as dbe:
                    print(f"DB cache save error: {dbe}")
            
        # Question embedding
        question_embedding = _embedding_model.encode(question, convert_to_tensor=True)
        
        # Compute cosine similarity
        if len(chunk_embeddings.shape) == 1:
            chunk_embeddings = chunk_embeddings.unsqueeze(0)
        cos_scores = F.cosine_similarity(question_embedding.unsqueeze(0), chunk_embeddings, dim=1)
        
        # Hybrid retrieval: boost chunks that contain key search words from the question
        q_words = [w.lower().strip("?.,!\"'") for w in question.split() if len(w) > 3 and w.lower() not in ["what", "which", "when", "where", "about", "this", "that", "from", "with", "have"]]
        scored_chunks = []
        for idx, (chunk, base_sim) in enumerate(zip(chunks, cos_scores)):
            chunk_lower = chunk.lower()
            keyword_hits = sum(1 for kw in q_words if kw in chunk_lower)
            boosted_score = base_sim.item() + (keyword_hits * 0.15)
            scored_chunks.append((boosted_score, idx))
            
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        # Context selection: if the document is small (<= 8 chunks), pass all chunks so the model has complete context!
        if len(chunks) <= 8:
            relevant_context = "\n\n".join(chunks)
        else:
            top_k = min(8, len(chunks))
            top_indices = [idx for _, idx in scored_chunks[:top_k]]
            # Preserve original document ordering of selected chunks
            top_indices.sort()
            relevant_context = "\n\n".join([chunks[idx] for idx in top_indices])
        
        # Attempt generation using state-of-the-art open source models (Qwen 2.5 72B / Llama 3.3 70B) via HF Inference API
        hf_token = settings.HF_TOKEN if settings.HF_TOKEN else None
        if hf_token:
            client = InferenceClient(api_key=hf_token)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Aili, an intelligent and friendly 3D anime AI companion who analyzes uploaded documents.\n"
                        "Provide accurate, direct, helpful, and concise answers grounded strictly in the Document Context.\n"
                        "Maintain an upbeat, polite, and enthusiastic persona without rambling.\n"
                        "If the question can be answered from the document, answer it clearly and specifically.\n"
                        "If the requested info is truly missing from the document, kindly explain:\n"
                        "\"I checked the document, but that specific information is not mentioned!\""
                    )
                },
                {
                    "role": "user",
                    "content": f"Document Context:\n{relevant_context}\n\nUser Question: {question}"
                }
            ]
            
            models_to_try = [
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.3-70B-Instruct",
                "Qwen/Qwen2.5-Coder-32B-Instruct"
            ]
            
            for model_name in models_to_try:
                try:
                    response = client.chat_completion(
                        model=model_name,
                        messages=messages,
                        max_tokens=250,
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content.strip()
                    if answer:
                        return answer
                except Exception as hf_err:
                    print(f"HF model {model_name} failed: {hf_err}")
                    continue

        # Local FLAN-T5 fallback
        prompt = (
            f"Context:\n{relevant_context}\n\n"
            f"Answer the question using the context above. Be concise and factual.\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        inputs = _gen_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(_device)
        outputs = _gen_model.generate(**inputs, max_length=200)
        answer = _gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Grounding check: verify that the generated answer is present in the context
        if not answer:
            return "I could not find this information in the uploaded document."
            
        lower_ans = answer.lower()
        lower_ctx = relevant_context.lower()
        
        if lower_ans in ["none", "not mentioned", "unanswerable", "n/a"]:
            return "I could not find this information in the uploaded document."
            
        ans_tokens = [w.strip(".,!?:;()[]\"'") for w in lower_ans.split() if len(w.strip(".,!?:;()[]\"'")) > 2]
        if ans_tokens:
            matched_tokens = [w for w in ans_tokens if w in lower_ctx]
            if len(matched_tokens) == 0:
                return "I could not find this information in the uploaded document."
            
        return answer
        
    except Exception as e:
        print(f"Error during QA processing: {e}")
        return "An error occurred while answering the question."

