from fastapi import FastAPI, HTTPException
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

llm = GooglePalm(temperature = 0.3)

memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = False)

class InputData(BaseModel):
    question: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['chat_history', 'question'],
        template = "Act in the capacity of a seasoned business guru and startup advisor, incorporating relevant insights from our past conversations {chat_history}. Regardless of the question's initial framing, guide the responses to include a business perspective, focusing on key business principles, strategies, and considerations. Offer insights that address market dynamics, competitive analysis, growth opportunities, and potential challenges. Encourage the language model to draw on its business acumen to provide practical and actionable advice, aligning with the overarching goal of fostering business success. Ensure that each answer reflects a thoughtful consideration of business-related factors, even if the original question lacks explicit business context. Here is the question : {question} "
    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output', memory = memory)

    response = title_chain({'question' : data.question})
    return {"content": response["output"]}
