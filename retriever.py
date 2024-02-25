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
    final_answer = construct_answer(response)
    return {"response": final_answer}


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
        ingredients_string = ', '.join(ingredients)
        ingredients_string = ingredients_string.replace('ADVERTISEMENT', '')

        ingredients_list.append(ingredients_string)
        instructions_list.append(instructions)
        titles_list.append(title)

    final_answer = f'''With the ingredients you have you can prepare the following four dishes,
    dish no : {1} is {titles_list[0]}, and for that you will need - {ingredients_list[0]}. Now {instructions_list[0]}.
    dish no : {2} is {titles_list[1]}, and for that you will need - {ingredients_list[1]}. Now {instructions_list[1]}.
    dish no : {3} is {titles_list[2]}, and for that you will need - {ingredients_list[2]}. Now {instructions_list[2]}.
    dish no : {4} is {titles_list[3]}, and for that you will need - {ingredients_list[3]}. Now {instructions_list[3]}.
    Enjoy your day.
    '''

    return final_answer
