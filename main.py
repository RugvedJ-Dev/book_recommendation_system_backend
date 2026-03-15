from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
app = FastAPI()

try:
    with open("models/top_50.pkl","rb") as f:
        load_fifty = pickle.load(f)
    with open("models/pivot_table.pkl","rb") as f:
        pivot_table = pickle.load(f)
    with open("models/sim_score.pkl","rb") as f:
        sim_score = pickle.load(f)
    with open("models/books_df.pkl","rb") as f:
        books_df = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Check your 'models/' directory.")

class Recommendation(BaseModel):
    book_name:str
    
@app.get("/")
def home():
    return {"message": "Server for book recommendation system"}

@app.get("/top_fifty")
async def top_fifty():
    df = pd.DataFrame(load_fifty)

    results = df.to_dict(orient='records')
    
    return results

@app.post("/get_recommendation")
async def get_recommendation(request: Recommendation):
    index = np.where(pivot_table.index == request.book_name)[0][0]
    similar_books = sorted(list(enumerate(sim_score[index])), key=lambda x: x[1], reverse=True)[1:11]
    
    data = []
    for i in similar_books:
        temp_df = books_df[books_df['Book-Title'] == pivot_table.index[i[0]]]
        book_info = temp_df.drop_duplicates("Book-Title").iloc[0]
        
        item = {
            "title": book_info["Book-Title"],
            "author": book_info["Book-Author"],
            "image_url": book_info["Image-URL-L"]
        }
        data.append(item)
        
    return data