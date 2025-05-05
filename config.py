import torch

# Data Preprocessing
PATCH_SIZE = (112, 112) # Standard size for face recognition models like ArcFace
IMG_SIZE = (112, 112) # Input image size if resizing is needed before landmark detection
# Define landmark indices for different facial regions (example using MediaPipe 468 landmarks)
# These indices would need careful selection based on the MediaPipe landmark map
LANDMARK_REGIONS = {
    "left_eye": [33, 160, 158, 133, 153, 144], # Example indices
    "right_eye": [362, 385, 387, 263, 373, 380], # Example indices
    "nose": [1, 4, 5, 6, 218, 438], # Example indices
    "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], # Example indices
    "jawline": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397], # Example indices
    # Add more regions as needed (e.g., eyebrows)
}
NUM_PATCHES = len(LANDMARK_REGIONS)

# Model
EMBEDDING_DIM = 512 # ArcFace typically outputs 512-D embeddings
ATTENTION_HEADS = 8
ATTENTION_DIM = 512 # Dimension after attention projection (can be same as EMBEDDING_DIM)
DROPOUT_RATE = 0.1

# Training
TRIPLET_MARGIN = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference
SIMILARITY_THRESHOLD = 0.6 # Example threshold for cosine similarity (adjust based on validation)
# For Euclidean distance, a different threshold would be needed

# Paths (Optional - can be passed as arguments)
# DATASET_PATH = "/path/to/your/triplet/dataset"
# MODEL_SAVE_PATH = "./faceformer_model.pth"
# ARCFACE_MODEL_PATH = "/path/to/pretrained/arcface.pth" # Or use a library function
