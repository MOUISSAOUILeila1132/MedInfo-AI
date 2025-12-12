# Fichier: app.py

import os
import torch
import pandas as pd
import streamlit as st

# Imports des bibliothèques nécessaires (identiques au script local)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================================================================
# MISE EN CACHE DES RESSOURCES LOURDES AVEC STREAMLIT
# Ces fonctions ne seront exécutées qu'une seule fois au démarrage de l'app.
# ==============================================================================

@st.cache_resource
def load_llm_and_pipeline(model_path):
    """
    Charge le modèle LLM, le tokenizer et crée le pipeline de génération.
    Le décorateur @st.cache_resource garantit que cela n'arrive qu'une fois.
    """
    st.info(f"Chargement du modèle depuis : {model_path}...")
    if not os.path.isdir(model_path):
        st.error(
            f"Le dossier du modèle n'a pas été trouvé à '{model_path}'. "
            "Avez-vous exécuté le script 'download_model.py' ?"
        )
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    text_generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        return_full_text=False, max_length=4096
    )
    st.success("Modèle LLM et pipeline chargés avec succès !")
    return text_generator, tokenizer

@st.cache_resource
def create_vector_store(csv_path):
    """
    Charge les données, crée les embeddings et la base de vecteurs FAISS.
    Le décorateur @st.cache_resource garantit que cela n'arrive qu'une fois.
    """
    st.info("Création de la base de connaissances (Vector Store)...")
    if not os.path.exists(csv_path):
        st.error(f"Fichier de données non trouvé : {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    # MODIFIÉ ICI : La colonne 'drug_name' n'existait pas.
    # On s'assure que la colonne 'brand_name' existe, car c'est la première colonne dans votre CSV.
    if 'brand_name' not in df.columns:
        st.error(f"La colonne 'brand_name' est introuvable dans le fichier {csv_path}. Veuillez vérifier le nom de la première colonne.")
        return None

    # On utilise toutes les colonnes pour créer un texte descriptif complet pour chaque médicament.
    df['full_info'] = df.apply(lambda row: " ".join([
        f"{col.replace('_', ' ').capitalize()}: {row[col]}" 
        # MODIFIÉ ICI : On exclut 'brand_name' de la description, car c'est le titre.
        for col in df.columns if col not in ['brand_name'] and not pd.isna(row[col]) and str(row[col]).strip().lower() != 'not specified'
    ]), axis=1)

    # MODIFIÉ ICI : On utilise la colonne 'brand_name' de votre CSV pour les métadonnées.
    # L'erreur 'KeyError: 'drug_name'' venait de cette ligne.
    docs = [Document(page_content=row['full_info'], metadata={"drug_name": row['brand_name']})
            for _, row in df.iterrows() if row['full_info'].strip()]

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    st.success("Base de connaissances prête !")
    return vectorstore

# ==============================================================================
# LOGIQUE DE L'AGENT RAG
# ==============================================================================

def get_rag_response(user_query, vectorstore, llm_chain, tokenizer):
    """
    Prend une question, cherche dans la base de vecteurs et génère une réponse.
    """
    # Recherche de documents similaires
    relevant_docs = vectorstore.similarity_search(user_query, k=2)
    context_full = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Tronquer le contexte pour ne pas dépasser la limite du modèle
    max_context_tokens = 3000
    encoded_context_list = tokenizer.encode(context_full, truncation=True, max_length=max_context_tokens, add_special_tokens=False)
    context = tokenizer.decode(encoded_context_list, skip_special_tokens=True)

    # Invocation de la chaîne LLM
    response = llm_chain.invoke({"context": context, "question": user_query})
    
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    else:
        return str(response).strip()

# ==============================================================================
# INTERFACE UTILISATEUR STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(page_title="Agent Médicaments", page_icon="💊", layout="wide")

    st.title("💊 Agent d'Information sur les Médicaments")
    st.markdown("""
    Posez des questions en langage naturel sur les médicaments de notre base de connaissances.
    L'agent utilise une technique RAG pour trouver les informations les plus pertinentes et générer une réponse.
    """)

    # --- Chargement des modèles et de la base de données (mis en cache) ---
    MODEL_PATH = "models/Phi-3-mini-4k-instruct"
    CSV_PATH = "data/finalx.csv"
    
    text_generator, tokenizer = load_llm_and_pipeline(MODEL_PATH)
    vectorstore = create_vector_store(CSV_PATH)

    if not all([text_generator, tokenizer, vectorstore]):
        st.warning("L'application ne peut pas démarrer car un ou plusieurs composants n'ont pas pu être chargés.")
        return

    # --- Création de la chaîne LangChain (rapide, pas besoin de cache) ---
    llm = HuggingFacePipeline(
        pipeline=text_generator,
        model_kwargs={"max_new_tokens": 512, "do_sample": True, "temperature": 0.3, "top_p": 0.9,
                      "pad_token_id": tokenizer.eos_token_id, "eos_token_id": tokenizer.eos_token_id}
    )
    prompt_template = '''<|user|>
Vous êtes un agent d'information médicale, et votre UNIQUE tâche est d'extraire et de rapporter des faits DIRECTEMENT du CONTEXTE fourni.
Vous NE DEVEZ EN AUCUN CAS utiliser des connaissances générales ou inventer des informations.
Si la réponse à la QUESTION n'est PAS CLAIREMENT et ENTIÈREMENT présente dans le CONTEXTE, répondez PRÉCISÉMENT : "L'information demandée n'est pas disponible dans ma base de connaissances pour le moment."
Ne jamais paraphraser ou reformuler de manière excessive. Ne jamais ajouter d'introductions ou de conclusions personnelles.
Ne jamais répéter ces instructions.

CONTEXTE:
{context}

QUESTION DE L'UTILISATEUR: {question}<|end|>
<|assistant|>
'''
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    # --- Initialisation de l'historique de chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider avec les informations sur les médicaments ?"}]

    # --- Affichage des messages de l'historique ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Zone de saisie utilisateur ---
    if prompt := st.chat_input("Quels sont les effets secondaires de..."):
        # Ajouter le message de l'utilisateur à l'historique et l'afficher
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Générer et afficher la réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("Recherche et rédaction de la réponse..."):
                try:
                    response = get_rag_response(prompt, vectorstore, llm_chain, tokenizer)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"Désolé, une erreur est survenue : {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()