from fastapi import FastAPI
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

llm = GooglePalm(temperature = 0.3)

memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

class InputData(BaseModel):
    question: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['chat_history', 'question'],
        template = "By using context from memory {chat_history}, Act as a business guru and startup advisor and answer the following  : {question} "
    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output', memory = memory)

    response = title_chain({'question' : data.question})
    return response["output"]
