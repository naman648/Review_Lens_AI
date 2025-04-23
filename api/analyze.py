from fastapi import FastAPI, Request
from serpapi import GoogleSearch
from langchain.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

@app.get("/analyze")
async def analyze_reviews(request: Request):
    place_id = request.query_params.get("place_id")
    if not place_id:
        return {"error": "Missing place_id"}

    params = {
        "engine": "google_maps_reviews",
        "data_id": place_id,
        "api_key": os.getenv("SERPAPI_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    reviews = [r["text"] for r in results.get("reviews", []) if "text" in r]

    if not reviews:
        return {"error": "No reviews found"}

    template = PromptTemplate(
        input_variables=["reviews"],
        template="""
        Here are customer reviews:\n{reviews}\n
        1. Summarize sentiment trend (positive/negative/neutral).
        2. List top 3 recurring themes.
        3. Suggest 2 improvements.
        """
    )

    # Using Groq 
    llm = ChatGroq(
        temperature=0.7,
        model="mixtral-8x7b-32768",  
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    chain = LLMChain(prompt=template, llm=llm)

    result = chain.run(reviews=" | ".join(reviews[:30]))
    return {"insights": result}