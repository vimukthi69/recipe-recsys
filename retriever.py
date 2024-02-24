from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
import configparser
from sentence_transformers import SentenceTransformer

# load configurations
config = configparser.ConfigParser()
config.read('config.conf')

# embedding model from hugging face
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# init pinecone
api_key = config.get('VECTORDB', 'API_KEY')
pc_index = config.get('VECTORDB', 'INDEX')
init = Pinecone(api_key=api_key)
index = init.Index(pc_index)

# init fastAPI
app = FastAPI()


# pydantic class for user query
class UserQuery(BaseModel):
    user_query: str


# POST call
@app.post("/completions/")
async def process_text(query_data: UserQuery):
    input_text = query_data.user_query

    response = pinecone_call(input_text)
    ingredients, instructions, titles = construct_answer(response)
    return {"titles": titles, "ingredients": ingredients, "instructions": instructions}


# pinecone query function
def pinecone_call(query):
    # generating embeddings for user query
    embeddings = model.encode(query)
    embeddings = embeddings.tolist()

    # pinecone query
    result = index.query(
                vector=embeddings,
                top_k=4,
                include_metadata=True
            )

    return result


# function to construct final answer
def construct_answer(json_response):
    ingredients_list = []
    instructions_list = []
    titles_list = []

    # Iterating over each 'matches' object
    for match in json_response['matches']:
        ingredients = match['metadata']['ingredients']
        instructions = match['metadata']['instructions']
        title = match['metadata']['title']

        # Add them to respective lists
        ingredients_list.append(ingredients)
        instructions_list.append(instructions)
        titles_list.append(title)

    return ingredients_list, instructions_list, titles_list
