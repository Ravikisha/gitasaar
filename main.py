import logging
from pathlib import Path
from typing import Optional
# import io

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
# from langchain.vectorstores import VectorStore
# from fastapi.responses import HTMLResponse,RedirectResponse

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# import json
# import os
# from ingest import ingest_docs

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore = None
con = sqlite3.connect('finalGita.db')
cur = con.cursor()
current_vectorstore = "gita_index"

embeddings = OpenAIEmbeddings()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/vectorstore", StaticFiles(directory="vectorstore"), name="vectorstore")

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore6.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    global vectorstore
    vectorstore = FAISS.load_local("gita_index", embeddings)
    print("vectorstore loaded")


# Pages link
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("indexpage.html",{"request": request})

@app.get("/gitapage")
async def get(request: Request):
    return templates.TemplateResponse("gitapage.html",{"request": request})
@app.get("/gitaanalysis")
async def get(request: Request):
    return templates.TemplateResponse("gitaanalysis.html",{"request": request})


# Gita ai part start

@app.get("/gita")
async def get(request: Request):
    return templates.TemplateResponse("gita.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            result = await qa_chain.ainvoke(
                {"question": question, "chat_history": chat_history}
            )
            # stream the result
            
            stream_resp = ChatResponse(sender="bot", message=result["answer"], type="stream")
            await websocket.send_json(stream_resp.dict())
            
            # print(result)
            # print(result["source_documents"][0]['metadata'])
            # docs1 =result["source_documents"]
            # print("*****************************************Source documents****************************************************")
            
            # print((result["source_documents"]))
            # print("****************************************Answer*****************************************************")
            # print(result["answer"])
            # print("*********************************************************************************************")
            sources_meta = []
            # for i in result['source_documents']:
            #     for row in cur.execute('SELECT Chapter,Verse,Devanagari,verseText,Translation FROM gita WHERE index = ?',(i.metadata,)):
            #         sources_meta.append(row[1])
            # for i in result['source_documents']:
            #     sources_meta.append(i.metadata['index'])
            
            for i in result['source_documents']:
                for row in cur.execute('SELECT Chapter,Verse,Devanagari,verseText,Translation FROM gita WHERE Chapter = ? AND Verse = ?',(i.metadata['Chapter'],i.metadata['Verse'])):
                    # print({"Chapter": row[0], "Verse": row[1], "Devanagari": row[2], "verseText": row[3], "Translation": row[4]})
                    sources_meta.append({"Chapter": row[0], "Verse": row[1], "Devanagari": row[2], "verseText": row[3], "Translation": row[4]})
                    # print(row)            
            
            chat_history.append((question, result["answer"]))
            end_resp = ChatResponse(sender="bot", message="", type="end")
            fin_data = end_resp.dict()
            # print(fin_data)
            # fin_data = {end_resp.dict(), "source_documents": sources_meta}
            fin_data = {**end_resp.dict(), **{"source_documents": sources_meta}}
            print(fin_data)
            await websocket.send_json(fin_data)
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
            
# gita part end

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
