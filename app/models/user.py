from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime
import re

PHONE_REGEX = re.compile(r"^\+?[1-9]\d{1,14}$")

class UserRegister(BaseModel):
    firstName: str = Field(..., min_length=1, description="First name of the user")
    lastName: str = Field(..., min_length=1, description="Last name of the user")
    email: EmailStr = Field(..., description="Email address of the user (must be unique)")
    phoneNo: str = Field(..., description="Phone number (10-15 digits format)")
    password: str = Field(..., min_length=8, max_length=72, description="Password for the account")
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if v.isdigit():
            raise ValueError('Password cannot be numbers only. It must include letters and numbers.')
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter.')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number.')
        return v
    
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    newPassword: str = Field(..., min_length=6, max_length=72)

class UserResponse(BaseModel):
    id: str
    firstName: str
    lastName: str
    email: EmailStr
    phoneNo: str
    createdAt: datetime
    
    class Config:
        populate_by_name = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str = Field(..., min_length=6, max_length=72)

class UpdateProfileRequest(BaseModel):
    firstName: Optional[str] = Field(None, min_length=1)
    lastName: Optional[str] = Field(None, min_length=1)
    phoneNo: Optional[str] = None
