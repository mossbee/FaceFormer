import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import EMBEDDING_DIM, NUM_PATCHES, ATTENTION_HEADS, ATTENTION_DIM, DROPOUT_RATE, DEVICE

# --- Placeholder for ArcFace Model ---
# In a real scenario, load a pre-trained ArcFace model (e.g., from insightface)
# and freeze its weights.
class ArcFaceEmbedder(nn.Module):
    def __init__(self, pretrained_path=None, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        # --- Replace this with actual ArcFace model loading ---
        # Example: using a simple CNN as a placeholder
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.embedding_dim)
        )
        # --- End Placeholder ---

        # Load pretrained weights if path is provided
        # if pretrained_path:
        #     self.model.load_state_dict(torch.load(pretrained_path))
        #     print(f"Loaded pretrained ArcFace weights from {pretrained_path}")

        # Freeze the ArcFace model
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval() # Set to evaluation mode

    def forward(self, patches):
        """
        Input: patches tensor of shape [B * N_patches, C, H, W] or [N_patches, C, H, W]
        Output: embeddings tensor of shape [B, N_patches, D] or [N_patches, D]
        """
        num_dims = patches.dim()
        if num_dims == 5: # Batch processing [B, N, C, H, W]
            B, N, C, H, W = patches.shape
            patches = patches.view(B * N, C, H, W)
            embeddings = self.model(patches)
            embeddings = embeddings.view(B, N, self.embedding_dim)
        elif num_dims == 4: # Single image processing [N, C, H, W]
            embeddings = self.model(patches) # Shape [N, D]
        else:
            raise ValueError(f"Unsupported input dimensions: {num_dims}")

        return embeddings

# --- Cross-Attention Block ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, num_heads=ATTENTION_HEADS, attention_dim=ATTENTION_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, "attention_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, attention_dim)
        self.k_proj = nn.Linear(embed_dim, attention_dim)
        self.v_proj = nn.Linear(embed_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, embed_dim) # Project back to original embed_dim
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize linear layers similar to Transformer implementations
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, query, key_value):
        """
        Input:
            query (Anchor embeddings): [B, N_patches_q, D] or [N_patches_q, D]
            key_value (Positive/Negative embeddings): [B, N_patches_kv, D] or [N_patches_kv, D]
        Output:
            attention_output: [B, N_patches_q, D] or [N_patches_q, D]
        """
        input_dim = query.dim()
        if input_dim == 2: # Add batch dimension if processing single pair
            query = query.unsqueeze(0)
            key_value = key_value.unsqueeze(0)

        B, N_q, D = query.shape
        B, N_kv, D_kv = key_value.shape
        assert D == D_kv, "Query and Key/Value must have the same embedding dimension"

        # Project Q, K, V
        q = self.q_proj(query) # [B, N_q, attention_dim]
        k = self.k_proj(key_value) # [B, N_kv, attention_dim]
        v = self.v_proj(key_value) # [B, N_kv, attention_dim]

        # Reshape for multi-head attention
        q = q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_q, head_dim]
        k = k.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_kv, head_dim]
        v = v.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, N_kv, head_dim]

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # [B, num_heads, N_q, N_kv]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = attn_probs @ v # [B, num_heads, N_q, head_dim]

        # Reshape and project back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N_q, attention_dim) # [B, N_q, attention_dim]
        output = self.out_proj(attn_output) # [B, N_q, D]

        if input_dim == 2: # Remove batch dimension if added
            output = output.squeeze(0)

        return output

# --- Main FaceFormer Model ---
class FaceFormerModel(nn.Module):
    def __init__(self, arcface_embedder, num_patches=NUM_PATCHES, embed_dim=EMBEDDING_DIM,
                 num_heads=ATTENTION_HEADS, attention_dim=ATTENTION_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.arcface_embedder = arcface_embedder
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Cross-attention block (trainable)
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads, attention_dim, dropout)

        # Optional: LayerNorm or additional processing layers after attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def _embed(self, patches_batch):
        """Helper to embed patches, handling potential batch dimension."""
        # patches_batch shape: [B, N_patches, C, H, W] or [N_patches, C, H, W]
        return self.arcface_embedder(patches_batch) # Output: [B, N_patches, D] or [N_patches, D]

    def _process_pair(self, embeds_a, embeds_b):
        """Processes one pair (e.g., Anchor-Positive) using cross-attention and fusion."""
        # embeds_a, embeds_b shape: [B, N, D] or [N, D]

        # Cross-attention: Attend from A (query) to B (key/value)
        attn_output_a = self.cross_attention(embeds_a, embeds_b)

        # Fusion: Additive fusion with LayerNorm (common practice)
        fused_a = self.norm1(embeds_a + attn_output_a)

        # Aggregate patch embeddings to a global representation (e.g., mean pooling)
        global_a = fused_a.mean(dim=-2) # Mean across patches dim -> [B, D] or [D]

        # Also compute global representation for B (without attention fusion for distance)
        global_b = embeds_b.mean(dim=-2)

        return global_a, global_b

    def forward(self, anchor_patches, positive_patches, negative_patches=None):
        """
        Forward pass for training (triplet) or inference (pair).

        Input (Training):
            anchor_patches: [B, N, C, H, W]
            positive_patches: [B, N, C, H, W]
            negative_patches: [B, N, C, H, W]
        Input (Inference):
            anchor_patches: [N, C, H, W] or [B, N, C, H, W]
            positive_patches: [N, C, H, W] or [B, N, C, H, W]
            negative_patches: None

        Output (Training):
            dist_ap: Euclidean distance between anchor and positive [B]
            dist_an: Euclidean distance between anchor and negative [B]
        Output (Inference):
            dist_ap: Euclidean distance between anchor and positive [B] or scalar
        """
        # 1. Embed all patches (ArcFace is frozen)
        embeds_a = self._embed(anchor_patches)
        embeds_p = self._embed(positive_patches)

        # 2. Process Anchor-Positive pair
        global_a_p, global_p = self._process_pair(embeds_a, embeds_p)
        dist_ap = F.pairwise_distance(global_a_p, global_p, p=2) # Euclidean distance

        if negative_patches is not None: # Training mode
            embeds_n = self._embed(negative_patches)
            # 3. Process Anchor-Negative pair
            global_a_n, global_n = self._process_pair(embeds_a, embeds_n)
            dist_an = F.pairwise_distance(global_a_n, global_n, p=2)
            return dist_ap, dist_an
        else: # Inference mode
            return dist_ap

    def predict_similarity(self, patches1, patches2):
        """
        Computes similarity score between two sets of patches for inference.
        Input:
            patches1: [N, C, H, W]
            patches2: [N, C, H, W]
        Output:
            similarity: Cosine similarity score (scalar)
            distance: Euclidean distance (scalar)
        """
        self.eval() # Ensure model is in eval mode
        with torch.no_grad():
            embeds1 = self._embed(patches1.to(DEVICE)) # [N, D]
            embeds2 = self._embed(patches2.to(DEVICE)) # [N, D]

            # Process pair (A->B attention)
            global1_att, global2_raw = self._process_pair(embeds1, embeds2)

            # Optionally, process pair (B->A attention) and average?
            # global2_att, global1_raw = self._process_pair(embeds2, embeds1)
            # final_global1 = (global1_att + global1_raw) / 2
            # final_global2 = (global2_att + global2_raw) / 2

            # Using simple A->B attention result for this example
            final_global1 = global1_att
            final_global2 = global2_raw # Use raw mean embedding for the second face

            distance = F.pairwise_distance(final_global1.unsqueeze(0), final_global2.unsqueeze(0), p=2).item()
            similarity = F.cosine_similarity(final_global1.unsqueeze(0), final_global2.unsqueeze(0)).item()

            return similarity, distance
