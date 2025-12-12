import os
import pandas as pd
import torch
from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login

# --- Configuration ---
HF_TOKEN = "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"
MODEL_ID = "google/flan-t5-large"
CSV_FILE_PATH = os.path.join("data", "drugs_data.csv")

app = Flask(__name__)

llm_chain = None
vectorstore = None
tokenizer = None

def init_agent():
    global llm_chain, vectorstore, tokenizer

    print("Connexion à Hugging Face Hub...")
    login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du périphérique : {device}")

    print(f"Chargement des données depuis {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH)

    # --- DÉBUT DE LA MODIFICATION MAJEURE ---
    # Au lieu de créer un seul gros document par médicament, nous en créons plusieurs petits.
    
    docs = []
    # Itérer sur chaque ligne (chaque médicament)
    for index, row in df.iterrows():
        drug_name = row['drug_name']
        # Itérer sur chaque colonne d'information (dosage, side_effects, etc.)
        for col in df.columns:
            # Ignorer la colonne du nom elle-même et les cellules vides
            if col != 'drug_name' and pd.notna(row[col]):
                # Créer un contenu de document ciblé et clair
                # Exemple : "Information regarding dosage for aspirin: Directions drink a full glass..."
                page_content = f"Information regarding {col.replace('_', ' ')} for {drug_name}: {row[col]}"
                
                # Créer un document LangChain avec des métadonnées utiles
                docs.append(Document(
                    page_content=page_content,
                    metadata={"drug_name": drug_name, "category": col}
                ))
    
    print(f"Préparé {len(docs)} documents ciblés à partir de {len(df)} médicaments.")
    # --- FIN DE LA MODIFICATION MAJEURE ---

    print("Chargement du modèle d'embeddings et création du VectorStore FAISS...")
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    print("VectorStore FAISS créé.")

    print(f"Chargement du modèle LLM : {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config
    )

    text_generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    llm_pipeline = HuggingFacePipeline(pipeline=text_generator)
    print("Pipeline LLM créé.")

    prompt_template = """
Basé UNIQUEMENT sur le contexte ci-dessous, répondez à la question.
Si la réponse ne se trouve pas dans le contexte, dites exactement : "L'information demandée n'est pas disponible dans ma base de connaissances."

Contexte :
{context}

Question :
{question}

Réponse :
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm_pipeline, prompt=PROMPT)
    print("Chaîne LLM créée. L'agent est prêt !")

# ... Le reste du fichier (ask_agent, home, etc.) ne change pas ...
def ask_agent(user_query: str) -> str:
    if not llm_chain or not vectorstore or not tokenizer:
        return "Erreur : L'agent n'est pas initialisé."

    print(f"Recherche de documents pour la requête : '{user_query}'")
    relevant_docs = vectorstore.similarity_search(user_query, k=2)
    
    # Juste pour le débogage, affichons ce qui a été trouvé
    print("\n--- Documents pertinents trouvés ---")
    for doc in relevant_docs:
        print(f"Source: {doc.metadata.get('drug_name')}, Catégorie: {doc.metadata.get('category')}")
        print(f"Contenu: {doc.page_content[:150]}...") # Affiche les 150 premiers caractères
    print("--------------------------------\n")
        
    context_full = "\n\n".join([doc.page_content for doc in relevant_docs])

    max_context_tokens = 3000
    encoded_context = tokenizer.encode(context_full, truncation=True, max_length=max_context_tokens)
    context = tokenizer.decode(encoded_context, skip_special_tokens=True)

    print("Génération de la réponse avec le LLM...")
    response = llm_chain.invoke({"context": context, "question": user_query})

    return response["text"] if isinstance(response, dict) and "text" in response else str(response)

@app.route('/', methods=['GET', 'POST'])
def home():
    question = ""
    reponse = ""
    if request.method == 'POST':
        question = request.form['question']
        if question:
            reponse = ask_agent(question)
    return render_template('index.html', question=question, reponse=reponse)

if __name__ == '__main__':
    init_agent()
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)