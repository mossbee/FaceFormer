import torch

from model import FaceFormerModel, ArcFaceEmbedder
from data_utils import preprocess_image
from config import (DEVICE, SIMILARITY_THRESHOLD, EMBEDDING_DIM, NUM_PATCHES,
                    ATTENTION_HEADS, ATTENTION_DIM, DROPOUT_RATE)

def load_model_for_inference(model_path, arcface_embedder):
    """Loads the trained FaceFormer model weights."""
    model = FaceFormerModel(
        arcface_embedder=arcface_embedder,
        num_patches=NUM_PATCHES,
        embed_dim=EMBEDDING_DIM,
        num_heads=ATTENTION_HEADS,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT_RATE # Dropout is typically disabled in eval mode anyway
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    print(f"Loaded trained model from {model_path}")
    return model

def verify_faces(image1_path, image2_path, model, threshold=SIMILARITY_THRESHOLD):
    """
    Verifies if two images belong to the same person using the trained model.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        model (FaceFormerModel): The loaded and trained model.
        threshold (float): The similarity threshold for matching.

    Returns:
        tuple: (is_same_person (bool), similarity_score (float), distance (float))
    """
    print(f"Verifying images: {image1_path} and {image2_path}")

    # 1. Preprocess both images
    patches1 = preprocess_image(image1_path)
    patches2 = preprocess_image(image2_path)

    if patches1 is None or patches2 is None:
        print("Error: Could not preprocess one or both images.")
        return False, 0.0, float('inf') # Indicate failure

    # 2. Move patches to the correct device
    patches1 = patches1.to(DEVICE)
    patches2 = patches2.to(DEVICE)

    # 3. Get similarity and distance from the model
    similarity, distance = model.predict_similarity(patches1, patches2)

    # 4. Compare similarity with threshold
    # Note: Using similarity here. If using distance, the comparison logic reverses.
    is_same_person = similarity >= threshold

    print(f"Similarity Score: {similarity:.4f}, Euclidean Distance: {distance:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Result: {'Same Person' if is_same_person else 'Different Persons'}")

    return is_same_person, similarity, distance

# Example Usage (requires a trained model checkpoint)
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_CHECKPOINT = "path/to/your/faceformer_final.pth" # IMPORTANT: Replace with your trained model path
    IMAGE1 = "path/to/personA_image1.jpg" # IMPORTANT: Replace with your test image
    IMAGE2 = "path/to/personA_image2.jpg" # IMPORTANT: Replace with your test image (same person)
    IMAGE3 = "path/to/personB_image1.jpg" # IMPORTANT: Replace with your test image (different person)

    # --- Initialization ---
    # Initialize the frozen ArcFace embedder (same as in training)
    # Ensure this matches the embedder used during training
    arcface_embedder = ArcFaceEmbedder().to(DEVICE)
    arcface_embedder.eval()

    # Load the trained FaceFormer model
    try:
        faceformer_model = load_model_for_inference(MODEL_CHECKPOINT, arcface_embedder)

        # --- Run Verification ---
        print("\n--- Verification Test 1 (Same Person) ---")
        verify_faces(IMAGE1, IMAGE2, faceformer_model)

        print("\n--- Verification Test 2 (Different Persons) ---")
        verify_faces(IMAGE1, IMAGE3, faceformer_model)

    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT}")
        print("Please train the model first or provide the correct path.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

