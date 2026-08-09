from fastapi import FastAPI, UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel
from transformers import pipeline
import whisper
import os

app = FastAPI()

vectorstore = None

# Load Models
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

whisper_model = whisper.load_model("base")

# For summary
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.
Use ONLY the information provided in the context below to answer the question.
If the answer is not available in the context, reply:
"I couldn't find the answer in the uploaded video."

Context:
{context}

Question:
{question}

Answer:
"""
)

# For Q & A
llm = ChatOllama(
    model = "qwen3:4b",
    temperature = 0.3,
    base_url = "http://host.docker.internal:11434"
)

# Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Question(BaseModel):
    question:str
    thread_id:str = "video_chat" 

@tool
def transcribe_video(filepath: str) -> str:
    """Transcribe video"""
    return whisper_model.transcribe(filepath)["text"]

@tool
def summarize_text(text:str) ->str:
    """Summarize given text"""
    result = summarizer(
        text,
        max_length=80,
        min_length=20,
        do_sample=False
    )
    return result[0]["summary_text"]

@tool
def search_video(question: str) -> str:
    """
    Search the uploaded video's transcript and return the
    most relevant context needed to answer the user's question.
    """

    global vectorstore

    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        doc.page_content
        for doc in docs
    )

    return context

@tool
def answer_question(data: dict) -> str:
    """Answer a question using the retrieved context."""

    prompt = qa_prompt.format(
        context=data["context"],
        question=data["question"]
    )

    result = llm.invoke(prompt)

    return result.content

    return result[0]["generated_text"]

Disk_memory = sqlite3.connect(
    "memory.db",
    check_same_thread= False
)

memory = SqliteSaver(Disk_memory)

# Wrap every tool in one
tools = [
    transcribe_video,
    summarize_text,
    search_video
]

agent = create_react_agent(
    model = llm,
    tools = tools,
    checkpointer = memory
)

# Home Route
@app.get("/")
def home():
    return {
        "message": "AI Video Summarizer API Running"
    }

# Upload Video
@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):

    # Save uploaded video
    filepath = os.path.join(UPLOAD_FOLDER, video.filename)

    with open(filepath, "wb") as f:
        f.write(await video.read())

    # Speech to Text

    transcript = transcribe_video.invoke({"filepath": filepath})

    # Chunk Transcript

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50

    )

    chunks = text_splitter.split_text(transcript)


    global vectorstore

    vectorstore = FAISS.from_texts(
    chunks,
    embedding_model
    )

    vectorstore.save_local("faiss_index")
    

    # Summarize Each Chunk
    partial_summaries = []

    for chunk in chunks:
        summary = summarize_text.invoke({"text":chunk})
        partial_summaries.append(summary)

    # Final Summary
    final_summary = " ".join(partial_summaries)

    return {
        "filename": video.filename,
        "transcript": transcript,
        "summary": final_summary,
        "chunks": len(chunks)
    }

# Q & A summary
@app.post('/get')
def ask(data:Question):

    global vectorstore

    if vectorstore is None:
        return {
            "error": "Upload a video first."
        }


    query = data.question

    response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    },
    config = {
        "configurable" :{
            "thread_id": data.thread_id
        }
    }
)

    return {
    "question": query,
    "answer": response["messages"][-1].content
}