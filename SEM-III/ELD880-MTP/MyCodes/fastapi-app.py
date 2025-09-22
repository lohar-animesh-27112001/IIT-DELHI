from fastapi import FastAPI
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/fact-data")
def read_fact_data():
    with open("./Data/cft_og_combined_data_sampled_gpt2_with_questions_downsampled_Fact Check.json", "r") as file:
        data = json.load(file)
    return data