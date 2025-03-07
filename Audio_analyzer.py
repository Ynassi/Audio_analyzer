import os
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from accelerate import Accelerator

# Chargement des variables d'environnement à partir du fichier .env
load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialiser Accelerate pour mieux gérer la mémoire
accelerator = Accelerator()

# Fonction pour transformer l'audio avec Whisper d'OpenAI (audio découpé en tranches de 30 secondes)
def transcript_audio(audio_file):
    # Utilisation d'un modèle Whisper plus léger
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",  # Utilisation de la version plus petite de Whisper
        chunk_length_s=30,
    )
    result = pipe(audio_file, batch_size=4)["text"]  # Réduire la taille du batch pour mieux gérer la mémoire
    return result

# Fonction pour générer un résumé avec Llama 2 (modèle plus léger)
def summarize_text(text):
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Modèle plus lourd
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Charger le modèle Llama 2 seulement en cas de besoin
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Utilisation de l'auto-distribution entre CPU et GPU
        offload_folder="path_to_offload",  # Déchargement du modèle si nécessaire
        torch_dtype=torch.float16,  # Utiliser float16 pour économiser de la mémoire
    )

    # Créer un pipeline de génération de texte pour résumer
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)  # Réduire max_length

    # Résumer le texte généré par Whisper
    summary = summarizer(text, max_length=100, do_sample=False)[0]["generated_text"]
    return summary

# Fonction principale qui combine la transcription et le résumé
def process_audio(audio_file):
    # Étape 1 : Transcription audio avec Whisper
    transcribed_text = transcript_audio(audio_file)
    
    # Étape 2 : Résumer le texte avec Llama 2
    summarized_text = summarize_text(transcribed_text)
    
    return summarized_text

# Configuration de l'interface Gradio
audio_input = gr.Audio(sources="upload", type="filepath")  # Input audio
output_text = gr.Textbox()  # Output text (résumé)

iface = gr.Interface(fn=process_audio,  # Utilisation de la fonction qui combine transcription et résumé
                     inputs=audio_input, 
                     outputs=output_text, 
                     title="Audio Transcription and Summarization App", 
                     description="Upload an audio file to transcribe and summarize it.") 

# Lancer le serveur Gradio
iface.launch(server_name="0.0.0.0", server_port=7860)
