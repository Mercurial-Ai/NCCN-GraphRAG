import json
from json import JSONDecodeError
import os
import re
from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set the API key and model name
MODEL = "gpt-4"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Neo4j connection (replace with your Neo4j credentials)
uri = "neo4j+s://410d4d72.databases.neo4j.io"
driver = GraphDatabase.driver(uri, auth=("neo4j", "giY0WFy5FMzrqnWaBhZx0_szsdJE9LEMwvhc010UHDk"))

def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")
    print("Database cleared successfully.")

def process_json(json_data):
    prompt = f"""
    Analyze the following JSON representing a clinical flowchart. 
    Extract all relevant entities and relationships based on the structure and content of the JSON.
    Be specific and detailed in identifying entity types and relationship types.
    Respond with a valid JSON object containing 'entities' and 'relationships' arrays.
    Each entity should have a 'name', 'type', and 'properties' (if any).
    Each relationship should have 'from', 'to', 'type', and 'properties' (if any).

    JSON to analyze:
    {json.dumps(json_data)}
    """
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in analyzing clinical flowcharts and extracting detailed entity-relationship information. Always respond with a valid JSON object without any markdown formatting."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response_content = completion.choices[0].message.content
    
    # Remove markdown code block if present
    response_content = re.sub(r'```json\s*|\s*```', '', response_content).strip()
    
    try:
        entities_relationships = json.loads(response_content)
    except JSONDecodeError:
        print(f"Error: Invalid JSON response from GPT model. Response: {response_content}")
        entities_relationships = {"entities": [], "relationships": []}
    
    if not isinstance(entities_relationships, dict) or 'entities' not in entities_relationships or 'relationships' not in entities_relationships:
        print(f"Error: Unexpected response structure. Response: {entities_relationships}")
        entities_relationships = {"entities": [], "relationships": []}

    return entities_relationships

def flatten_properties(properties):
    flattened = {}
    for key, value in properties.items():
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (list, dict)):
            flattened[key] = json.dumps(value)
    return flattened

def create_mini_graph(tx, entities_relationships, title):
    print(f"Creating mini graph for title: {title}")
    
    for entity in entities_relationships['entities']:
        entity_name = entity.get('name')
        entity_type = entity.get('type')
        properties = entity.get('properties', {})
        
        # Ensure properties is a dictionary and flatten complex values
        if not isinstance(properties, dict):
            properties = {}
        properties = flatten_properties(properties)
        
        # Create a dynamic label based on the entity type
        label = ''.join(word.capitalize() for word in entity_type.split())
        
        # Create the node with its properties
        tx.run(f"MERGE (e:{label} {{name: $name}}) "
               "SET e += $properties", 
               name=entity_name, properties=properties)
    
    for rel in entities_relationships.get('relationships', []):
        from_entity = rel.get('from')
        to_entity = rel.get('to')
        rel_type = rel.get('type', '').upper().replace(' ', '_')
        properties = rel.get('properties', {})
        
        # Ensure properties is a dictionary and flatten complex values
        if not isinstance(properties, dict):
            properties = {}
        properties = flatten_properties(properties)
        
        # Create the relationship with its properties
        tx.run(
            f"MATCH (a {{name: $from_entity}}), (b {{name: $to_entity}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += $properties",
            from_entity=from_entity, to_entity=to_entity, properties=properties
        )
    
    # Create the Title node and connect it to all entities
    tx.run("MERGE (t:Title {name: $title})", title=title)
    
    for entity in entities_relationships['entities']:
        entity_name = entity.get('name')
        tx.run(
            "MATCH (t:Title {name: $title}), (e {name: $entity_name}) "
            "MERGE (t)-[:CONTAINS]->(e)",
            title=title, entity_name=entity_name
        )

    print(f"Created/Merged graph for title: {title}")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def combine_graphs_semantic(titles):
    print("Combining graphs using semantic similarity...")
    
    # Get embeddings for all titles
    title_embeddings = {title: get_embedding(title) for title in titles}
    
    with driver.session() as session:
        for i, title1 in enumerate(titles):
            for title2 in titles[i+1:]:
                similarity = compute_similarity(title_embeddings[title1], title_embeddings[title2])
                
                if similarity > 0.5:  # You can adjust this threshold
                    session.run(
                        "MATCH (t1:Title {name: $title1}), (t2:Title {name: $title2}) "
                        "MERGE (t1)-[r:SEMANTICALLY_RELATED]->(t2) "
                        "SET r.similarity = $similarity",
                        title1=title1, title2=title2, similarity=float(similarity)
                    )
                    print(f"Created SEMANTICALLY_RELATED relationship between '{title1}' and '{title2}' with similarity {similarity}")

def process_jsonl_file(file_path):
    titles = []
    print(f"Processing file: {file_path}")

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            print(f"Processing line {i}")
            try:
                json_data = json.loads(line)
                title = json_data.get('title')
                if not title:
                    print(f"Skipping line {i} due to missing title")
                    continue
                titles.append(title)
                
                entities_relationships = process_json(json_data)
                
                with driver.session() as session:
                    try:
                        session.execute_write(create_mini_graph, entities_relationships, title)
                        print(f"Completed processing for title: {title}")
                    except Exception as e:
                        print(f"Error creating graph for title '{title}': {str(e)}")
                        continue
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error processing line {i}: {str(e)}")
                continue
    
    print(f"Processed {len(titles)} titles")    
    return titles

def create_index(tx):
    tx.run("CREATE INDEX IF NOT EXISTS FOR (t:Title) ON (t.name)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity)")

# Main execution
file_path = "NCCN JSON Data/nccn_json_data.jsonl"
try:
    with driver.session() as session:
        session.execute_write(clear_database)
    
    titles = process_jsonl_file(file_path)
    with driver.session() as session:
        session.execute_write(create_index)
    combine_graphs_semantic(titles)
    print("Knowledge Graph creation completed successfully.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    driver.close()