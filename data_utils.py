import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from config import PATCH_SIZE, LANDMARK_REGIONS, IMG_SIZE

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def detect_landmarks(image_np):
    """Detects 468 face landmarks using MediaPipe Face Mesh."""
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    # Assuming only one face is detected
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image_np.shape
    landmarks_coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    return landmarks_coords # Shape: (468, 2)

def get_region_center(landmarks, region_indices):
    """Calculates the center of a landmark region."""
    region_landmarks = landmarks[region_indices]
    center = np.mean(region_landmarks, axis=0)
    return center.astype(int)

def extract_patch(image_np, center_xy, patch_size):
    """Extracts a square patch centered at center_xy."""
    cx, cy = center_xy
    h, w, _ = image_np.shape
    ph, pw = patch_size

    # Calculate patch boundaries, handling image edges
    x1 = max(0, cx - pw // 2)
    y1 = max(0, cy - ph // 2)
    x2 = min(w, cx + (pw + 1) // 2) # Adjust for odd sizes
    y2 = min(h, cy + (ph + 1) // 2) # Adjust for odd sizes

    patch = image_np[y1:y2, x1:x2]

    # If patch is smaller due to boundaries, pad it
    pad_left = max(0, pw // 2 - cx)
    pad_top = max(0, ph // 2 - cy)
    pad_right = max(0, (cx + (pw + 1) // 2) - w)
    pad_bottom = max(0, (cy + (ph + 1) // 2) - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize to the target patch size
    patch_resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
    return patch_resized

# Define standard normalization for face models
normalize_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Or use ImageNet stats if ArcFace expects them
])

def preprocess_image(image_path):
    """Loads an image, detects landmarks, extracts patches, and normalizes them."""
    try:
        image = Image.open(image_path).convert('RGB')
        # Optional: Resize image before landmark detection if needed
        # image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

        landmarks = detect_landmarks(image_np)
        if landmarks is None:
            print(f"Warning: No face detected in {image_path}")
            return None # Handle cases where no face is found

        all_patches = []
        for region_name, indices in LANDMARK_REGIONS.items():
            center = get_region_center(landmarks, indices)
            patch_np = extract_patch(image_np, center, PATCH_SIZE)
            # Convert patch back to RGB for ToTensor and normalize
            patch_rgb = cv2.cvtColor(patch_np, cv2.COLOR_BGR2RGB)
            patch_pil = Image.fromarray(patch_rgb)
            patch_tensor = normalize_transform(patch_pil)
            all_patches.append(patch_tensor)

        if not all_patches:
            return None

        # Stack patches into a single tensor: [N_patches, C, H, W]
        patches_tensor = torch.stack(all_patches)
        return patches_tensor

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Example Usage (for testing)
# if __name__ == "__main__":
#     test_image_path = "path/to/your/test_face.jpg"
#     patches = preprocess_image(test_image_path)
#     if patches is not None:
#         print(f"Extracted {patches.shape[0]} patches with shape: {patches.shape}")
#         # You could save/visualize patches here
#         # import torchvision
#         # torchvision.utils.save_image(patches, "patches_grid.png", nrow=len(LANDMARK_REGIONS))
#     else:
#         print("Preprocessing failed.")
