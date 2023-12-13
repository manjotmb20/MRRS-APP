import json
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from langdetect import detect
from googletrans import Translator
import stanza
from collections import defaultdict

# Loading and setting up language models
nlp_en = spacy.load("en_core_web_sm")  # English model
nlp_es = spacy.load("es_core_news_sm") # Spanish model
nlp_hi = spacy.load("xx_ent_wiki_sm")  # Hindi model (using multilingual model)

stanza.download('hi')

# Function to detect language
def detect_language(query):
    try:
        return detect(query)
    except Exception as e:
        return "Error: " + str(e)

# Function for advanced query parsing for Hindi
def parse_hindi_query(query):
    nlp_hi = stanza.Pipeline(lang='hi', processors='tokenize,pos,lemma')
    doc = nlp_hi(query)
    return [word.lemma for sent in doc.sentences for word in sent.words]

# Function to parse query based on detected language
def process_query(query):
    lang = detect_language(query)
    if "Error" in lang:
        return lang

    if lang == 'hi':
        keywords = parse_hindi_query(query)
    elif lang == 'en':
        doc = nlp_en(query)
    elif lang == 'es':
        doc = nlp_es(query)
    else:
        return "Unsupported language"

    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return {"language": lang, "keywords": keywords}

# Function to translate query
def translate_query(query, target_language):
    translator = Translator()
    translation = translator.translate(query, dest=target_language)
    return translation.text

# Function to get translated queries in multiple languages
def get_translated_queries(user_query):
    lang = detect_language(user_query)
    if lang == "hi":
        translated_queries = {
            'hindi': user_query,
            'spanish': translate_query(user_query, 'es'),
            'english': translate_query(user_query, 'en')
        }
    elif lang == "es":
        translated_queries = {
            'hindi': translate_query(user_query, 'hi'),
            'spanish': user_query,
            'english': translate_query(user_query, 'en')
        }
    elif lang == "en":
        translated_queries = {
            'hindi': translate_query(user_query, 'hi'),
            'spanish': translate_query(user_query, 'es'),
            'english': user_query
        }
    else:
        translated_queries = {
            'hindi': translate_query(user_query, 'hi'),
            'spanish': translate_query(user_query, 'es'),
            'english': translate_query(user_query, 'en')
        }
    return translated_queries

# Load recipe JSON data
with open('recipes.json', 'r') as file:
    recipes = json.load(file)

# Embedding function for text
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text)

# Function to embed a recipe
def embed_recipe(recipe):
    combined_text = f"{recipe['recipeName']} {' '.join(recipe['ingredients'])} {' '.join(recipe['instruction'])}"
    return embed_text(combined_text)

# Embedding recipes and building FAISS index
database_embeddings = [embed_recipe(recipe) for recipe in recipes]
dim = len(database_embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(database_embeddings).astype('float32'))

# Function to search the database
def search_database(query_embedding, k=5):
    _, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return [recipes[i] for i in indices[0]]

# Function to calculate ingredient similarity
def calculate_ingredient_similarity(recipe_ingredients, query_embedding):
    recipe_ingredients_str = ' '.join(recipe_ingredients)
    recipe_emb = model.encode(recipe_ingredients_str, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(recipe_emb, query_embedding).item()
    return similarity_score

# Function to calculate recipe alignment
def calculate_recipe_alignment(recipe_instructions, query_embedding):
    recipe_instructions_str = ' '.join(recipe_instructions)
    recipe_emb = model.encode(recipe_instructions_str, convert_to_tensor=True)
    alignment_score = util.pytorch_cos_sim(recipe_emb, query_embedding).item()
    return alignment_score

# Function to calculate the overall score for a recipe
def calculate_score(recipe, query_embedding):
    recipe_embedding = embed_recipe(recipe)
    semantic_similarity = util.pytorch_cos_sim(recipe_embedding, query_embedding).item()
    ingredient_similarity = calculate_ingredient_similarity(recipe['ingredients'], query_embedding)
    language_match = 1 if detect_language(recipe['recipeName']) == detect_language(query_embedding) else 0
    recipe_alignment = calculate_recipe_alignment(recipe['instruction'], query_embedding)
    score = (semantic_similarity * 0.5) + (ingredient_similarity * 0.2) + (language_match * 0.2) + (recipe_alignment * 0.1)
    return score

# Function to search and score recipes
def search_and_score_recipes(query):
    translated_queries = get_translated_queries(query)
    scored_results = defaultdict(list)
    for lang, translated_query in translated_queries.items():
        query_embedding = embed_text(translated_query)
        search_results = search_database(query_embedding)
        for recipe in search_results:
            score = calculate_score(recipe, query_embedding)
            scored_results[lang].append((recipe, score))

    for lang, results in scored_results.items():
        scored_results[lang] = sorted(results, key=lambda x: x[1], reverse=True)
    return scored_results
