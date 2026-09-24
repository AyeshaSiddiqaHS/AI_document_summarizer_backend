import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Configuration from your settings
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "ai_document_summarizer"

async def init_db():
    print("Connecting to MongoDB...")
    try:
        client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        
        # Ping the server to verify connection
        await client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        db = client[DATABASE_NAME]
        
        # In MongoDB, databases and collections are created automatically when you insert the first document.
        # However, we can force creation of indexes right now so the database exists before you run the app.
        
        print("Ensuring unique index on 'email' field in 'users' collection...")
        await db.users.create_index("email", unique=True)
        
        print(f"Database '{DATABASE_NAME}' is initialized and ready to use!")
        
    except Exception as e:
        print("Failed to connect to MongoDB. Ensure the service is running.")
        print(f"Error details: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(init_db())
