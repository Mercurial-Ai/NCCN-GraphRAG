from neo4j import GraphDatabase
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up Neo4j connection
uri = "neo4j+s://410d4d72.databases.neo4j.io"
driver = GraphDatabase.driver(uri, auth=("neo4j", "giY0WFy5FMzrqnWaBhZx0_szsdJE9LEMwvhc010UHDk"))

# Set up OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def find_similar_titles(question, top_n=5):
    question_embedding = get_embedding(question)
    
    with driver.session() as session:
        result = session.run("MATCH (t:Title) RETURN t.name AS title")
        titles = [record["title"] for record in result]
    
    # Calculate embeddings for all titles
    title_embeddings = [get_embedding(title) for title in titles]
    
    # Calculate similarities
    similarities = cosine_similarity([question_embedding], title_embeddings)[0]
    
    # Get top N similar titles
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [titles[i] for i in top_indices]

def get_related_info(title):
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Title {name: $title})-[:CONTAINS]->(e)
            RETURN e.name AS entity_name, labels(e) AS entity_types, properties(e) AS properties
        """, title=title)
        return [dict(record) for record in result]

def answer_question(question):
    similar_titles = find_similar_titles(question)
    related_info = []
    for title in similar_titles:
        related_info.extend(get_related_info(title))
    
    context = f"""
    Question: {question}

    Relevant Information:
    Similar Titles: {', '.join(similar_titles)}

    Related Entities:
    {related_info}

    Based on this information, please provide a detailed and accurate answer to the question.
    If the information is not sufficient to answer the question completely, please state so and provide the best possible answer with the available information.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant providing detailed and accurate answers to questions about cancer treatments based on data from a knowledge graph."},
            {"role": "user", "content": context}
        ]
    )
    
    return response.choices[0].message.content.strip()

# User interface
while True:
    user_question = input("What's your Question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    
    try:
        answer = answer_question(user_question)
        print(f"\nAnswer: {answer}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Close the Neo4j driver
driver.close()