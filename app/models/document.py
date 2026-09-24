from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentResponse(BaseModel):
    id: str
    filename: str
    fileFormat: str
    fileSize: int
    filePath: str
    userId: str
    createdAt: datetime
    updatedAt: datetime
    isFavorite: bool = False
    summary: Optional[str] = None
    summaryShort: Optional[str] = None
    summaryMedium: Optional[str] = None
    summaryLong: Optional[str] = None
    keyInsights: Optional[str] = None

