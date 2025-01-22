import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
import random

# =============================================================================
#                            MODEL DEFINITION
# =============================================================================
class SmallRLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SmallRLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.manifold = PoincareBall(c=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.manifold.expmap0(x)  # Map to hyperbolic space
        x = self.fc2(self.manifold.logmap0(x))  # Back to Euclidean space
        return x

class SmallerRLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SmallerRLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
#                            REWARD FUNCTIONS
# =============================================================================
def compute_accuracy_reward(output, target):
    """Reward based on cosine similarity between output and target embeddings."""
    cos_sim = F.cosine_similarity(output, target, dim=-1)
    return cos_sim.mean().item()

def compute_format_reward(response, required_format):
    """Reward for adhering to the required format."""
    if required_format in response:
        return 1.0
    return -1.0

def compute_combined_reward(output, target, response, required_format):
    """Combine accuracy and format rewards."""
    accuracy_reward = compute_accuracy_reward(output, target)
    format_reward = compute_format_reward(response, required_format)
    return accuracy_reward + format_reward

# =============================================================================
#                            TRAINING FUNCTION
# =============================================================================
def train_with_rl(
    model,
    optimizer,
    dataset,
    input_dim,
    output_dim,
    epochs=5,
    max_seq_len=50
):
    history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for sample in dataset:
            # Extract prompt and target embedding (dummy data here for simplicity)
            prompt = sample["Prompt"]
            target_text = sample["Target"]

            # Convert text to embeddings (random here for demo purposes)
            input_embedding = torch.rand((1, max_seq_len, input_dim)).to(device)
            target_embedding = torch.rand((1, max_seq_len, output_dim)).to(device)

            optimizer.zero_grad()

            # Forward pass
            output_embedding = model(input_embedding)

            # Decode output embedding to simulate model response
            model_response = "<think> Simulated response </think>"  # Simulated decoding for demo purposes

            # Compute rewards
            required_format = "<think>"
            combined_reward = compute_combined_reward(
                output_embedding, target_embedding, model_response, required_format
            )

            # Loss = negative reward (maximize reward)
            loss = -torch.tensor(combined_reward, requires_grad=True).to(device)

            loss.backward()
            optimizer.step()

            history.append({
                "epoch": epoch + 1,
                "prompt": prompt,
                "reward": combined_reward,
                "loss": loss.item(),
                "response": model_response
            })

            print(f"Prompt: {prompt[:30]}... | Response: {model_response} | Combined Reward: {combined_reward:.4f} | Loss: {loss.item():.4f}")

    return history

# =============================================================================
#                      DISTILLATION FUNCTION
# =============================================================================
def distill_model(teacher_model, student_model, dataset, optimizer, epochs=5, max_seq_len=50):
    """Distill knowledge from the teacher model to the smaller student model."""
    for epoch in range(epochs):
        print(f"Distillation Epoch {epoch + 1}/{epochs}")
        for sample in dataset:
            # Extract prompt (dummy data here for simplicity)
            prompt = sample["Prompt"]

            # Convert text to embeddings (random here for demo purposes)
            input_embedding = torch.rand((1, max_seq_len, input_dim)).to(device)

            # Teacher model output
            with torch.no_grad():
                teacher_output = teacher_model(input_embedding)

            # Student model output
            student_output = student_model(input_embedding)

            # Loss = MSE between teacher and student outputs
            loss = F.mse_loss(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Prompt: {prompt[:30]}... | Distillation Loss: {loss.item():.4f}")

# =============================================================================
#                            MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # ---------------------------
    # 1) Device Setup
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 2) Model and Optimizer Setup
    # ---------------------------
    input_dim = 128  # Example input dimension
    hidden_dim = 256  # Hidden layer dimension
    output_dim = 128  # Output dimension matches input for reconstruction

    model = SmallRLModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-4)

    # ---------------------------
    # 3) Dataset (Dummy Data)
    # ---------------------------
    # Replace with actual reasoning task data
    dataset = [
        {"Prompt": "Solve x + 2 = 5.", "Target": "x = 3."},
        {"Prompt": "What is the capital of France?", "Target": "Paris."},
        {"Prompt": "Explain Pythagoras' theorem.", "Target": "a^2 + b^2 = c^2 for a right triangle."}
    ]

    # ---------------------------
    # 4) Training
    # ---------------------------
    history = train_with_rl(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=10,
        max_seq_len=50
    )

    print("Training complete.")

    # ---------------------------
    # 5) Distillation
    # ---------------------------
    smaller_model = SmallerRLModel(input_dim, hidden_dim // 2, output_dim).to(device)
    distill_optimizer = torch.optim.Adam(smaller_model.parameters(), lr=1e-4)

    distill_model(
        teacher_model=model,
        student_model=smaller_model,
        dataset=dataset,
        optimizer=distill_optimizer,
        epochs=5,
        max_seq_len=50
    )

    print("Distillation complete.")
