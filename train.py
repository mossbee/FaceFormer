import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Assuming you have a custom dataset

from model import FaceFormerModel, ArcFaceEmbedder
from loss import TripletLoss
from data_utils import preprocess_image # Used within dataset
from config import (DEVICE, LEARNING_RATE, EPOCHS, BATCH_SIZE,
                    TRIPLET_MARGIN, EMBEDDING_DIM, NUM_PATCHES,
                    ATTENTION_HEADS, ATTENTION_DIM, DROPOUT_RATE)

# --- Placeholder Dataset ---
# Replace with your actual dataset loading triplets (anchor_path, positive_path, negative_path)
class TripletFaceDataset(Dataset):
    def __init__(self, triplet_list_file):
        # self.triplets = load_triplet_paths_from_file(triplet_list_file)
        self.triplets = [
            ("path/to/anchor1.jpg", "path/to/positive1.jpg", "path/to/negative1.jpg"),
            ("path/to/anchor2.jpg", "path/to/positive2.jpg", "path/to/negative2.jpg"),
            # ... more triplets
        ] # Placeholder

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor_patches = preprocess_image(anchor_path)
        positive_patches = preprocess_image(positive_path)
        negative_patches = preprocess_image(negative_path)

        # Handle cases where preprocessing might fail
        if anchor_patches is None or positive_patches is None or negative_patches is None:
            print(f"Warning: Skipping triplet index {idx} due to preprocessing error.")
            # Return None or handle appropriately in collate_fn or training loop
            # For simplicity, we might return zero tensors, but skipping is better.
            # This requires a custom collate_fn to filter None values.
            # As a simple placeholder, we return dummy data, but this should be fixed.
            dummy_patch = torch.zeros((NUM_PATCHES, 3, *PATCH_SIZE))
            return dummy_patch, dummy_patch, dummy_patch

        return anchor_patches, positive_patches, negative_patches

# Custom collate_fn to filter out None samples
def collate_fn(batch):
    # Filter out None entries caused by preprocessing errors
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if not batch:
        return None # Return None if the whole batch is invalid

    # Unzip the batch
    anchor_patches, positive_patches, negative_patches = zip(*batch)

    # Stack tensors
    anchor_patches = torch.stack(anchor_patches)
    positive_patches = torch.stack(positive_patches)
    negative_patches = torch.stack(negative_patches)

    return anchor_patches, positive_patches, negative_patches

# --- Main Training Function ---
def train():
    print(f"Using device: {DEVICE}")

    # 1. Dataset and DataLoader
    # train_dataset = TripletFaceDataset("path/to/your/train_triplets.txt")
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # Placeholder loader
    print("Warning: Using placeholder data loader.")
    train_dataset = TripletFaceDataset(None) # Pass None for placeholder
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


    # 2. Model Initialization
    # Initialize ArcFace embedder (load pretrained weights here if applicable)
    arcface_embedder = ArcFaceEmbedder().to(DEVICE)
    arcface_embedder.eval() # Ensure it's frozen and in eval mode

    # Initialize the main model
    model = FaceFormerModel(
        arcface_embedder=arcface_embedder,
        num_patches=NUM_PATCHES,
        embed_dim=EMBEDDING_DIM,
        num_heads=ATTENTION_HEADS,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    # 3. Loss Function
    criterion = TripletLoss(margin=TRIPLET_MARGIN)

    # 4. Optimizer (only optimize trainable parameters - CrossAttention, LayerNorms, etc.)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode (enables dropout, etc.)
        running_loss = 0.0
        processed_batches = 0

        for i, batch_data in enumerate(train_loader):
            if batch_data is None: # Skip batch if collate_fn returned None
                print(f"Skipping empty batch {i+1}")
                continue

            anchor_patches, positive_patches, negative_patches = batch_data
            anchor_patches = anchor_patches.to(DEVICE)
            positive_patches = positive_patches.to(DEVICE)
            negative_patches = negative_patches.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            dist_ap, dist_an = model(anchor_patches, positive_patches, negative_patches)

            # Calculate loss
            loss = criterion(dist_ap, dist_an)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

            if (i + 1) % 10 == 0: # Print progress every 10 batches
                 if processed_batches > 0:
                    print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / processed_batches:.4f}")


        epoch_loss = running_loss / processed_batches if processed_batches > 0 else 0.0
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {epoch_loss:.4f}")

        # Optional: Add validation loop here
        # Optional: Save model checkpoint
        # torch.save(model.state_dict(), f"faceformer_epoch_{epoch+1}.pth")

    print("Training finished.")
    # Optional: Save final model
    # torch.save(model.state_dict(), "faceformer_final.pth")

if __name__ == "__main__":
    # Note: You need a dataset of image triplets and potentially a pretrained ArcFace model.
    # The placeholder dataset and ArcFace model need to be replaced.
    train()
