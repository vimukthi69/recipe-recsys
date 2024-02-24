import json
import configparser
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# load configurations
config = configparser.ConfigParser()
config.read('config.conf')

# pinecone initialization
api_key = config.get('VECTORDB', 'API_KEY')
pc_index = config.get('VECTORDB', 'INDEX')
init = Pinecone(api_key=api_key)
index = init.Index(pc_index)

# embedding model from hugging face
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# track data points
count = 1

with open('contents/recipes_raw_nosource_ar.json', 'r') as file:

    # loading json dataset
    data = json.load(file)

    # extracting details from each json object
    for key, value in data.items():
        title = value.get('title', '')
        ingredients = value.get('ingredients', [])
        instructions = value.get('instructions', '')

        # data preprocessing
        ingredients_string = ', '.join(ingredients)
        ingredients_string = ingredients_string.replace('ADVERTISEMENT', '')
        ingredients_string = 'Ingredients are : ' + ingredients_string

        # checking for null ingredients
        if len(ingredients) == 0:
            continue

        else:
            embeddings = model.encode(ingredients_string)
            # converting numpy array to a list
            embeddings = embeddings.tolist()

            # inserting vectors to pinecone
            index.upsert(
                vectors=[
                    {
                        "id": str(count),
                        "values": embeddings,
                        "metadata": {"title": title, "ingredients": ingredients, "instructions": instructions}
                    }
                ]
            )
            print('vector - ', count, ' inserted successfully')
            count += 1
