"""
LifeOS Database Models and Configuration
WORKING VERSION: Uses bcrypt directly, no passlib dependency issues
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import bcrypt

# Database URL with fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lifeos.db")

# Create engine with proper settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    """User model with secure password hashing using bcrypt directly"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)
    
    # Relationships
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """
        Hash and set the password using bcrypt directly
        Handles passwords of any length by truncating to 72 bytes
        """
        # Encode password to bytes
        password_bytes = password.encode('utf-8')
        
        # Truncate to 72 bytes if necessary (bcrypt's limit)
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
        
        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        # Store as string
        self.password_hash = hashed.decode('utf-8')
    
    def verify_password(self, password: str) -> bool:
        """
        Verify password against hash using bcrypt directly
        """
        try:
            # Encode password to bytes
            password_bytes = password.encode('utf-8')
            
            # Truncate to 72 bytes if necessary (same as set_password)
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
            
            # Get stored hash as bytes
            stored_hash = self.password_hash.encode('utf-8')
            
            # Verify
            return bcrypt.checkpw(password_bytes, stored_hash)
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class ChatHistory(Base):
    """Chat history model with proper indexing"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    agent_id = Column(String(50), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'agent'
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_history")
    
    # Composite index for common queries
    __table_args__ = (
        Index('idx_user_agent_timestamp', 'user_id', 'agent_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<ChatHistory(id={self.id}, user_id={self.user_id}, agent='{self.agent_id}')>"


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def create_db_and_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


def get_db():
    """Dependency function to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_default_user(db):
    """Create a default test user if none exist"""
    try:
        if db.query(User).count() == 0:
            test_user = User(username="testuser")
            test_user.set_password("testpass123")
            db.add(test_user)
            db.commit()
            print("✅ Default test user created (username: testuser, password: testpass123)")
    except Exception as e:
        print(f"❌ Error creating default user: {e}")
        db.rollback()