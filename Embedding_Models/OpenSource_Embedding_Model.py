from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3"
)

# Documents
text = [
    "To bake a chocolate cake, begin by preheating the oven to 180°C. In a large bowl, mix flour, cocoa powder, baking powder, and sugar. In a separate bowl, whisk together eggs, milk, and vegetable oil. Combine the wet and dry ingredients until smooth. Pour the batter into a greased cake pan and bake for approximately 30–35 minutes. Allow the cake to cool before adding chocolate frosting and serving.",

    "Making chocolate frosting requires butter, cocoa powder, powdered sugar, and a small amount of milk. Beat the butter until creamy, gradually add the dry ingredients, and mix until smooth. The frosting can be spread over cakes, cupcakes, or brownies to provide a rich chocolate flavor.",

    "To prepare a vanilla cake, combine flour, sugar, baking powder, eggs, milk, and vanilla extract. Mix until smooth and pour the batter into a baking pan. Bake in a preheated oven until golden brown. Allow the cake to cool completely before decorating with frosting or fruit toppings.",

    "Pasta can be prepared by boiling water and adding dried noodles. Once cooked, the pasta is drained and mixed with a sauce such as tomato, pesto, or alfredo. Additional ingredients like vegetables, cheese, or meat can be added to create a complete meal.",

    "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data. Common algorithms include decision trees, neural networks, and support vector machines. Machine learning is widely used in recommendation systems, fraud detection, and computer vision."
]

text_embeddings = embeddings.embed_documents(text)

query = "How do I bake a chocolate cake?"

query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], text_embeddings)[0]


best_index = np.argmax(scores)

print("Index:", best_index)
print("Content:", text[best_index])
print("Similarity Score:", scores[best_index])