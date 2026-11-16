"""Evaluation of a LoRA fine tuned Mistral 7B on medical question answering."""

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "samurvivor-07/medical-mistral-lora"
DATASET = "medalpaca/medical_meadow_medical_flashcards"
SPLIT_SEED = 42
