import os
import pandas as pd
import torch
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
# Ligne correcte
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
from transformers import BitsAndBytesConfig

# --- Configuration ---
# Il est FORTEMENT recommandé d'utiliser des variables d'environnement pour les secrets.
# Pour ce test, nous utilisons directement votre token.
HF_TOKEN = "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"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
CSV_FILE_PATH = os.path.join("data", "drugs_data.csv")

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Variables globales pour l'agent (chargées une seule fois) ---
llm_chain = None
vectorstore = None
tokenizer = None

def init_agent():
    """
    Charge les modèles, le tokenizer, et prépare la base de données vectorielle.
    Cette fonction est appelée une seule fois au démarrage du serveur.
    """
    global llm_chain, vectorstore, tokenizer

    # 1. Connexion à Hugging Face
    print("Connexion à Hugging Face Hub...")
    login(token=HF_TOKEN)

    # 2. Vérification du GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du périphérique : {device}")

    # 3. Chargement et traitement des données
    print(f"Chargement des données depuis {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH)
    df['full_info'] = df.apply(lambda row: " ".join([
        f"{col.replace('_', ' ').capitalize()}: {row[col]}"
        for col in df.columns if col not in ['drug_name'] and not pd.isna(row[col])
    ]), axis=1)
    docs = [Document(page_content=row['full_info'], metadata={"drug_name": row['drug_name']}) for _, row in df.iterrows()]
    print(f"{len(docs)} documents préparés.")

    # 4. Création de la base de vecteurs
    print("Chargement du modèle d'embeddings et création du VectorStore FAISS...")
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    print("VectorStore FAISS créé.")

    # 5. Chargement du LLM et du Tokenizer
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # ou load_in_4bit=True pour encore moins de mémoire
    bnb_4bit_compute_dtype=torch.bfloat16
)
    print(f"Chargement du modèle LLM : {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    # 6. Création du pipeline de génération de texte
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        return_full_text=False
    )
    llm_pipeline = HuggingFacePipeline(
        pipeline=text_generator,
        model_kwargs={"temperature": 0.3, "top_p": 0.9}
    )
    print("Pipeline LLM créé.")

    # 7. Création du Prompt et de la chaîne
    prompt_template = """<|user|>
Vous êtes un agent d'information médicale. Votre unique tâche est de rapporter des faits directement du CONTEXTE fourni.
Si la réponse n'est pas clairement dans le CONTEXTE, répondez : "L'information demandée n'est pas disponible dans ma base de connaissances."
Ne jamais inventer d'informations.

CONTEXTE:
{context}

QUESTION: {question}<|end|>
<|assistant|>
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm_pipeline, prompt=PROMPT)
    print("Chaîne LLM créée. L'agent est prêt !")

def ask_agent(user_query: str) -> str:
    """
    Exécute une requête de l'utilisateur contre l'agent RAG.
    """
    if not llm_chain or not vectorstore or not tokenizer:
        return "Erreur : L'agent n'est pas initialisé."

    print(f"Recherche de documents pour la requête : '{user_query}'")
    # Recherche de documents pertinents
    relevant_docs = vectorstore.similarity_search(user_query, k=2)
    context_full = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Troncature du contexte pour éviter les dépassements de mémoire
    max_context_tokens = 3000
    encoded_context = tokenizer.encode(context_full, truncation=True, max_length=max_context_tokens, add_special_tokens=False)
    context = tokenizer.decode(encoded_context, skip_special_tokens=True)

    print("Génération de la réponse avec le LLM...")
    response = llm_chain.invoke({"context": context, "question": user_query})

    return response["text"] if isinstance(response, dict) and "text" in response else str(response)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Gère les requêtes GET pour afficher la page et POST pour traiter une question.
    """
    question = ""
    reponse = ""
    if request.method == 'POST':
        question = request.form['question']
        if question:
            reponse = ask_agent(question)

    return render_template('index.html', question=question, reponse=reponse)

if __name__ == '__main__':
    init_agent()  # Initialiser l'agent avant de démarrer le serveur
    app.run(host='0.0.0.0', port=5000, debug=True)
