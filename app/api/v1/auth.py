from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import EmailStr
from datetime import datetime, timezone, timedelta
from jose import jwt, JWTError
from bson import ObjectId

from app.models.user import (
    UserRegister, UserLogin, ForgotPasswordRequest, 
    ResetPasswordRequest, UserResponse, TokenResponse,
    ChangePasswordRequest, UpdateProfileRequest
)
from app.core.security import (
    hash_password, verify_password, create_access_token, generate_reset_token
)
from app.core.config import settings
from app.core.database import get_database

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    db = get_database()
    
    # Check if token is blacklisted
    blacklisted = await db["token_blacklist"].find_one({"token": token})
    if blacklisted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = await db["users"].find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    db = get_database()
    
    # Check if user already exists
    existing_user = await db["users"].find_one({"email": user_data.email.lower()})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    now = datetime.now(timezone.utc)
    new_user = {
        "firstName": user_data.firstName,
        "lastName": user_data.lastName,
        "email": user_data.email.lower(),
        "phoneNo": user_data.phoneNo,
        "hashedPassword": hash_password(user_data.password),
        "resetPasswordToken": None,
        "resetPasswordExpires": None,
        "createdAt": now,
        "updatedAt": now
    }
    
    result = await db["users"].insert_one(new_user)
    
    # Ensure index exists (usually done at startup, but safe to include)
    await db["users"].create_index("email", unique=True)
    
    return UserResponse(
        id=str(result.inserted_id),
        firstName=new_user["firstName"],
        lastName=new_user["lastName"],
        email=new_user["email"],
        phoneNo=new_user["phoneNo"],
        createdAt=new_user["createdAt"]
    )

@router.post("/login", response_model=TokenResponse)
async def login(credentials: OAuth2PasswordRequestForm = Depends()):
    db = get_database()
    user = await db["users"].find_one({"email": credentials.username.lower()})
    
    if not user or not verify_password(credentials.password, user["hashedPassword"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return TokenResponse(access_token=access_token)

@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    db = get_database()
    user = await db["users"].find_one({"email": req.email.lower()})
    if not user:
        # Don't reveal if email exists or not
        return {"message": "If that email exists, a reset link will be sent."}
        
    reset_token = generate_reset_token()
    reset_expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {
            "resetPasswordToken": reset_token,
            "resetPasswordExpires": reset_expires
        }}
    )
    
    # In a real app, send an email here. We'll just print for local dev.
    print(f"\n[DEV MODE] Password Reset Link: http://localhost:5173/reset-password?token={reset_token}\n")
    
    return {"message": "If that email exists, a reset link will be sent."}

@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    db = get_database()
    
    user = await db["users"].find_one({
        "resetPasswordToken": req.token,
        "resetPasswordExpires": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    new_hashed_password = hash_password(req.newPassword)
    
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {
            "hashedPassword": new_hashed_password,
            "resetPasswordToken": None,
            "resetPasswordExpires": None,
            "updatedAt": datetime.now(timezone.utc)
        }}
    )
    
    return {"message": "Password successfully reset"}

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    db = get_database()
    # Add token to blacklist so it can't be used again
    await db["token_blacklist"].insert_one({
        "token": token,
        "blacklistedAt": datetime.now(timezone.utc)
    })
    
    # Optional: cleanup old tokens (e.g. older than 24h) to prevent collection from growing indefinitely
    expiry_limit = datetime.now(timezone.utc) - timedelta(days=1)
    await db["token_blacklist"].delete_many({"blacklistedAt": {"$lt": expiry_limit}})
    
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=str(current_user["_id"]),
        firstName=current_user["firstName"],
        lastName=current_user["lastName"],
        email=current_user["email"],
        phoneNo=current_user["phoneNo"],
        createdAt=current_user["createdAt"]
    )

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    req: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user)
):
    db = get_database()
    update_data = {}
    if req.firstName is not None and req.firstName.strip():
        update_data["firstName"] = req.firstName.strip()
    if req.lastName is not None and req.lastName.strip():
        update_data["lastName"] = req.lastName.strip()
    if req.phoneNo is not None:
        update_data["phoneNo"] = req.phoneNo.strip()
        
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided to update.")

    update_data["updatedAt"] = datetime.now(timezone.utc)

    await db["users"].update_one(
        {"_id": current_user["_id"]},
        {"$set": update_data}
    )

    updated_user = await db["users"].find_one({"_id": current_user["_id"]})

    return UserResponse(
        id=str(updated_user["_id"]),
        firstName=updated_user["firstName"],
        lastName=updated_user["lastName"],
        email=updated_user["email"],
        phoneNo=updated_user.get("phoneNo", ""),
        createdAt=updated_user["createdAt"]
    )

@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest, 
    current_user: dict = Depends(get_current_user)
):
    db = get_database()
    
    # Verify current password
    if not verify_password(req.currentPassword, current_user["hashedPassword"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
        
    # Hash new password
    new_hashed = hash_password(req.newPassword)
    
    # Update DB
    await db["users"].update_one(
        {"_id": current_user["_id"]},
        {"$set": {
            "hashedPassword": new_hashed,
            "updatedAt": datetime.now(timezone.utc)
        }}
    )
    
    return {"message": "Password updated successfully"}
