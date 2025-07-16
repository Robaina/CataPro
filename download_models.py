import os
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

# --- Configuration ---
# A dictionary mapping the Hugging Face model name to its class
MODELS_TO_DOWNLOAD = {
    "Rostlab/prot_t5_xl_uniref50": T5Model,
    "laituan245/molt5-base-smiles2caption": T5ForConditionalGeneration
}

# The local directory where you want to save the models
# This will create a 'models' folder in the same directory where you run the script.
SAVE_DIRECTORY = "models"

# --- Main Download Script ---
def download_model(model_name, model_class, save_dir):
    """
    Downloads a model and its tokenizer from Hugging Face and saves them
    in a subdirectory named after the model.
    """
    # Create a specific subdirectory for the current model
    model_save_path = os.path.join(save_dir, model_name.split('/')[-1])
    print(f"Preparing to download '{model_name}' into '{model_save_path}'...")

    try:
        # Download tokenizer and model from Hugging Face
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)

        # Save the tokenizer and model files to the specified local path
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained(model_save_path)

        print(f"Successfully downloaded and saved '{model_name}'.")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred while downloading {model_name}: {e}")

if __name__ == "__main__":
    print("Starting model download process...")
    # Create the main 'models' directory if it doesn't already exist
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    for name, model_cls in MODELS_TO_DOWNLOAD.items():
        download_model(name, model_cls, SAVE_DIRECTORY)

    print("\nModel download complete!")
    print(f"Your models are now available in the '{SAVE_DIRECTORY}' directory.")