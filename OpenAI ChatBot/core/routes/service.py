from typing import Union
from fastapi import APIRouter
from pydantic import BaseModel
from core.controllers.ingest_build import *

route = APIRouter()

class Item(BaseModel):
    angka1: float
    angka2: float
    
class QAItem(BaseModel):
    question : str

@route.get("/")
async def read_root():
    return {"Hello": "World"}

@route.post("/")
async def question_answer(item:QAItem):
    question = item.question
    qa = llm_chain.predict(question=question, context=page_content_string)
    return {"answer" : qa}



