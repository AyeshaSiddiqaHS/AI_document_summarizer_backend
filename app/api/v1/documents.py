from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from datetime import datetime, timezone
import os
import shutil
import mimetypes

from app.api.v1.auth import get_current_user
from app.core.database import get_database
from app.models.document import DocumentResponse

router = APIRouter()

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats are: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save file to disk
    # To avoid naming collisions, we prefix with timestamp and user id
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = f"{current_user['_id']}_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save file to disk."
        )
        
    file_size = os.path.getsize(file_path)
    now = datetime.now(timezone.utc)
    
    # Save metadata to DB
    db = get_database()
    new_doc = {
        "filename": file.filename,
        "fileFormat": ext.replace('.', '').upper(),
        "fileSize": file_size,
        "filePath": file_path,
        "userId": str(current_user["_id"]),
        "createdAt": now,
        "updatedAt": now,
        "isFavorite": False
    }
    
    result = await db["documents"].insert_one(new_doc)
    
    return DocumentResponse(
        id=str(result.inserted_id),
        filename=new_doc["filename"],
        fileFormat=new_doc["fileFormat"],
        fileSize=new_doc["fileSize"],
        filePath=new_doc["filePath"],
        userId=new_doc["userId"],
        createdAt=new_doc["createdAt"],
        updatedAt=new_doc["updatedAt"],
        isFavorite=new_doc["isFavorite"]
    )

from typing import List

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(current_user: dict = Depends(get_current_user)):
    db = get_database()
    cursor = db["documents"].find({"userId": str(current_user["_id"])}).sort("createdAt", -1)
    docs = await cursor.to_list(length=100)
    
    return [
        DocumentResponse(
            id=str(doc["_id"]),
            filename=doc["filename"],
            fileFormat=doc["fileFormat"],
            fileSize=doc["fileSize"],
            filePath=doc["filePath"],
            userId=doc["userId"],
            createdAt=doc["createdAt"],
            updatedAt=doc["updatedAt"],
            isFavorite=doc.get("isFavorite", False),
            summary=doc.get("summary"),
            summaryShort=doc.get("summaryShort"),
            summaryMedium=doc.get("summaryMedium"),
            summaryLong=doc.get("summaryLong"),
            keyInsights=doc.get("keyInsights")
        ) for doc in docs
    ]

from bson import ObjectId

@router.patch("/{doc_id}/favorite", response_model=DocumentResponse)
async def toggle_favorite(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    new_status = not doc.get("isFavorite", False)
    
    await db["documents"].update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"isFavorite": new_status, "updatedAt": datetime.now(timezone.utc)}}
    )
    
    updated_doc = await db["documents"].find_one({"_id": ObjectId(doc_id)})
    
    return DocumentResponse(
        id=str(updated_doc["_id"]),
        filename=updated_doc["filename"],
        fileFormat=updated_doc["fileFormat"],
        fileSize=updated_doc["fileSize"],
        filePath=updated_doc["filePath"],
        userId=updated_doc["userId"],
        createdAt=updated_doc["createdAt"],
        updatedAt=updated_doc["updatedAt"],
        isFavorite=updated_doc.get("isFavorite", False),
        summary=updated_doc.get("summary"),
        keyInsights=updated_doc.get("keyInsights")
    )

from app.core.summarizer import generate_summary, ask_document_question, generate_key_insights
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

@router.post("/{doc_id}/insights", response_model=DocumentResponse)
async def get_or_generate_insights(doc_id: str, force: bool = False, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if doc.get("keyInsights") and not force:
        pass
    else:
        insights = await generate_key_insights(str(doc["_id"]), doc["filePath"], doc["fileFormat"])
        if insights in [
            "Failed to read document.", 
            "No text available to extract key insights.",
            "This document is a scanned image or empty without extractable text. Please upload a document with readable text (PDF, DOCX, or TXT)."
        ]:
            raise HTTPException(
                status_code=400, 
                detail="This document does not contain extractable text (e.g. scanned image). Please upload a document with searchable text to extract insights."
            )
            
        await db["documents"].update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"keyInsights": insights, "updatedAt": datetime.now(timezone.utc)}}
        )
        doc["keyInsights"] = insights
        
    return DocumentResponse(
        id=str(doc["_id"]),
        filename=doc["filename"],
        fileFormat=doc["fileFormat"],
        fileSize=doc["fileSize"],
        filePath=doc["filePath"],
        userId=doc["userId"],
        createdAt=doc["createdAt"],
        updatedAt=doc["updatedAt"],
        isFavorite=doc.get("isFavorite", False),
        summary=doc.get("summary"),
        summaryShort=doc.get("summaryShort"),
        summaryMedium=doc.get("summaryMedium") or doc.get("summary"),
        summaryLong=doc.get("summaryLong"),
        keyInsights=doc.get("keyInsights")
    )

@router.post("/{doc_id}/summarize", response_model=DocumentResponse)
async def summarize_document(
    doc_id: str, 
    force: bool = False, 
    length: str = "medium", 
    current_user: dict = Depends(get_current_user)
):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    length_norm = (length or "medium").lower()
    if length_norm not in ["short", "medium", "long"]:
        length_norm = "medium"

    summary_field_map = {
        "short": "summaryShort",
        "medium": "summaryMedium",
        "long": "summaryLong"
    }
    target_field = summary_field_map[length_norm]
    
    existing_len_summary = doc.get(target_field) or (doc.get("summary") if length_norm == "medium" else None)

    if existing_len_summary and not force:
        # Already summarized for this requested length
        summary = existing_len_summary
    else:
        # Generate summary with the requested length
        summary = await generate_summary(str(doc["_id"]), doc["filePath"], doc["fileFormat"], length=length_norm)
        if summary in [
            "Failed to read document.", 
            "No text available to summarize in this document.",
            "An error occurred while generating the summary."
        ]:
            raise HTTPException(
                status_code=400, 
                detail="This document does not contain extractable text (e.g. scanned image). Please upload a document with searchable text to generate a summary."
            )
        
        update_dict = {
            target_field: summary,
            "summary": summary, # Keep default 'summary' synced with latest generated
            "updatedAt": datetime.now(timezone.utc)
        }
        
        await db["documents"].update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update_dict}
        )
        doc.update(update_dict)
        
    return DocumentResponse(
        id=str(doc["_id"]),
        filename=doc["filename"],
        fileFormat=doc["fileFormat"],
        fileSize=doc["fileSize"],
        filePath=doc["filePath"],
        userId=doc["userId"],
        createdAt=doc["createdAt"],
        updatedAt=doc["updatedAt"],
        isFavorite=doc.get("isFavorite", False),
        summary=doc.get("summary"),
        summaryShort=doc.get("summaryShort"),
        summaryMedium=doc.get("summaryMedium"),
        summaryLong=doc.get("summaryLong"),
        keyInsights=doc.get("keyInsights")
    )

@router.get("/{doc_id}/chat")
async def get_chat_history(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    history = await db["chat_history"].find(
        {"doc_id": str(doc_id), "userId": str(current_user["_id"])}
    ).sort("createdAt", 1).to_list(100)

    result = []
    for h in history:
        result.append({
            "id": str(h["_id"]),
            "role": h.get("role"),
            "content": h.get("content"),
            "createdAt": h.get("createdAt")
        })
    return result

@router.delete("/{doc_id}/chat")
async def clear_chat_history(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    await db["chat_history"].delete_many(
        {"doc_id": str(doc_id), "userId": str(current_user["_id"])}
    )
    return {"message": "Chat history cleared successfully"}

@router.post("/{doc_id}/chat")
async def chat_document(doc_id: str, req: ChatRequest, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
        
    user_msg = req.question.strip()
    # Save user message to chat history
    now = datetime.now(timezone.utc)
    await db["chat_history"].insert_one({
        "doc_id": str(doc_id),
        "userId": str(current_user["_id"]),
        "role": "user",
        "content": user_msg,
        "createdAt": now
    })

    answer = await ask_document_question(str(doc["_id"]), doc["filePath"], doc["fileFormat"], user_msg)
    
    if answer in ["Failed to read document.", "An error occurred while answering the question."]:
        raise HTTPException(status_code=500, detail=answer)
        
    # Save AI response to chat history
    await db["chat_history"].insert_one({
        "doc_id": str(doc_id),
        "userId": str(current_user["_id"]),
        "role": "ai",
        "content": answer,
        "createdAt": datetime.now(timezone.utc)
    })

    return {"answer": answer}

@router.delete("/{doc_id}/summary")
async def delete_document_summary(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Unset all summary variations and key insights from the document
    await db["documents"].update_one(
        {"_id": ObjectId(doc_id)},
        {
            "$unset": {
                "summary": "",
                "summaryShort": "",
                "summaryMedium": "",
                "summaryLong": "",
                "keyInsights": ""
            },
            "$set": {
                "updatedAt": datetime.now(timezone.utc)
            }
        }
    )
    
    return {"message": "Document summaries and insights deleted successfully"}

@router.delete("/{doc_id}")
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Delete file from disk
    if os.path.exists(doc["filePath"]):
        try:
            os.remove(doc["filePath"])
        except Exception as e:
            print(f"Error removing file {doc['filePath']}: {e}")
            
    # Delete chunks cache from DB
    await db["document_chunks"].delete_one({"doc_id": str(doc_id)})
            
    # Delete from DB
    await db["documents"].delete_one({"_id": ObjectId(doc_id)})
    
    return {"message": "Document deleted successfully"}

@router.get("/{doc_id}/download")
async def download_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    file_path = doc.get("filePath")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File on disk not found")
        
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "application/octet-stream"
        
    return FileResponse(
        path=file_path,
        filename=doc.get("filename", "document"),
        media_type=content_type,
        content_disposition_type="attachment"
    )

@router.get("/{doc_id}/file")
async def get_document_file(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    file_path = doc.get("filePath")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File on disk not found")
        
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "application/octet-stream"
        
    # For PDF and TXT, browser can display inline directly
    return FileResponse(
        path=file_path,
        filename=doc.get("filename", "document"),
        media_type=content_type,
        content_disposition_type="inline"
    )

@router.get("/{doc_id}/content")
async def get_document_content(doc_id: str, current_user: dict = Depends(get_current_user)):
    db = get_database()
    doc = await db["documents"].find_one({"_id": ObjectId(doc_id), "userId": str(current_user["_id"])})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    file_path = doc.get("filePath")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File on disk not found")
        
    try:
        from app.core.summarizer import extract_text_from_file
        text = extract_text_from_file(file_path, doc.get("fileFormat", ""))
        return {
            "id": str(doc["_id"]),
            "filename": doc.get("filename"),
            "fileFormat": doc.get("fileFormat"),
            "fileSize": doc.get("fileSize"),
            "content": text or "Document is empty or no extractable text found."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read document content: {str(e)}")


