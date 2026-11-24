from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.routers import chat, isl, image, session

app = FastAPI(
    title="Synora Backend",
    description="Multimodal Indian language companion backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(isl.router)
app.include_router(image.router)
app.include_router(session.router)


@app.get("/")
def root():
    return {"message": "Synora backend is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}