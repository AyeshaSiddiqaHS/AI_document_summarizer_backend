import asyncio
from app.core.summarizer import extract_text_from_file, generate_summary, ask_document_question
import os

def test():
    # create a dummy txt file
    dummy_file = "uploads/dummy_test.txt"
    os.makedirs("uploads", exist_ok=True)
    with open(dummy_file, "w") as f:
        f.write("My name is Jane Smith and my Aadhaar number is 9876 5432 1098. I love machine learning.")
        
    print("Testing Summarization...")
    summary = generate_summary(dummy_file, "TXT")
    print(f"Summary: {summary}")
    
    print("\nTesting QA - Found case")
    ans1 = ask_document_question(dummy_file, "TXT", "What is Jane's Aadhaar number?")
    print(f"Answer 1: {ans1}")
    
    print("\nTesting QA - Not Found case")
    from sentence_transformers import util
    import torch
    from app.core.summarizer import _embedding_model
    q_emb = _embedding_model.encode("What is the secret code to the moon?", convert_to_tensor=True)
    c_emb = _embedding_model.encode("My name is Jane Smith and my Aadhaar number is 9876 5432 1098. I love machine learning.", convert_to_tensor=True)
    print(f"Cosine Score: {util.cos_sim(q_emb, c_emb)[0].item()}")
    
    ans2 = ask_document_question(dummy_file, "TXT", "What is the secret code to the moon?")
    print(f"Answer 2: {ans2}")
    
    # cleanup
    os.remove(dummy_file)

if __name__ == "__main__":
    test()
