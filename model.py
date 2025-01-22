import random
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from difflib import SequenceMatcher

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PFAF dataset and initialize model
dataset = load_dataset("TuringsSolutions/PFAF750")
checkpoint = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Assign a padding token if it doesn't exist
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to accommodate new token

# Optimizer for fine-tuning
optimizer = AdamW(model.parameters(), lr=5e-5)

# Text Normalization Function
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip().lower()

# Fuzzy Matching Function
def is_similar(response, correct_response, threshold=0.9):
    normalized_response = normalize_text(response)
    normalized_correct_response = normalize_text(correct_response)
    similarity = SequenceMatcher(None, normalized_response, normalized_correct_response).ratio()
    return similarity >= threshold

# Feedback Generator
def feedback_generator(prompt, model_response, correct_response):
    if is_similar(model_response, correct_response):
        return f"Correct! The answer is indeed: {correct_response}."
    else:
        return (f"Incorrect. The correct answer is: {correct_response}. "
                f"Your response was: {model_response.strip()}.")

# Fine-tuning Function
def fine_tune_model(prompt, correct_response):
    model.train()  # Set model to training mode
    combined_text = prompt + tokenizer.eos_token + correct_response
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Create labels: ignore the prompt tokens, only compute loss on the correct_response tokens
    prompt_length = len(tokenizer(prompt + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100  # Set labels for the prompt tokens to -100

    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# Generate Backward Question
def generate_backward_question(correct_response):
    model.eval()
    backward_prompt = f"Reverse the logic: Given the solution '{correct_response}', reconstruct the question."
    inputs = tokenizer(backward_prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Fine-tune Backward Reasoning
def fine_tune_backward(backward_question, correct_backward_response):
    model.train()  # Set model to training mode
    combined_text = backward_question + tokenizer.eos_token + correct_backward_response
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Create labels: ignore the backward_question tokens, only compute loss on the correct_backward_response tokens
    backward_question_length = len(tokenizer(backward_question + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
    labels = input_ids.clone()
    labels[:, :backward_question_length] = -100  # Set labels for the backward_question tokens to -100

    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# Construct Prompt Function
def construct_prompt(example, default_prompt="Provide a detailed explanation for this response:"):
    if 'Prompt' not in example or example['Prompt'] is None:
        return f"{default_prompt}\n{example.get('Response', '')}"
    return example['Prompt']

# Adaptive Boundary Experiment with Reverse Thinking
def adaptive_boundary_experiment_with_reverse(dataset, epochs=1, max_length=512):
    results = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n" + "=" * 50)

        data_split = dataset['train']
        for example in data_split:
            prompt = construct_prompt(example)
            correct_response = example.get('Response', None)

            if correct_response is None:
                print(f"Skipping example due to missing response: {example}")
                continue

            print(f"Prompt: {prompt}")

            # Generate forward reasoning
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200)
            model_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"Model Response (Forward): {model_response}")

            # Generate backward question
            backward_question = generate_backward_question(correct_response)
            print(f"Backward Question: {backward_question}")

            # Generate backward reasoning
            backward_inputs = tokenizer(backward_question, return_tensors="pt", truncation=True, padding=True)
            backward_input_ids = backward_inputs["input_ids"].to(device)
            backward_attention_mask = backward_inputs["attention_mask"].to(device)
            with torch.no_grad():
                backward_outputs = model.generate(input_ids=backward_input_ids, attention_mask=backward_attention_mask, max_new_tokens=200)
            backward_response = tokenizer.decode(backward_outputs[0], skip_special_tokens=True).strip()
            print(f"Backward Reasoning: {backward_response}")

            # Consistency check between forward and backward reasoning
            is_consistent = is_similar(backward_response, prompt)
            print(f"Consistency Check: {'Passed' if is_consistent else 'Failed'}")

            # Generate feedback
            feedback = feedback_generator(prompt, model_response, correct_response)
            print(f"Feedback: {feedback}")

            # Fine-tune on incorrect responses
            if not is_similar(model_response, correct_response):
                loss = fine_tune_model(prompt, correct_response)
                print(f"Fine-tuned on forward reasoning. Loss: {loss:.4f}")

            if not is_consistent:
                backward_loss = fine_tune_backward(backward_question, prompt)
                print(f"Fine-tuned on backward reasoning. Loss: {backward_loss:.4f}")

            # Store results
            results.append({
                "epoch": epoch + 1,
                "prompt": prompt,
                "model_response": model_response,
                "backward_question": backward_question,
                "backward_response": backward_response,
                "correct_response": correct_response,
                "feedback": feedback,
                "forward_similarity": SequenceMatcher(None, normalize_text(model_response), normalize_text(correct_response)).ratio(),
                "backward_consistency": is_consistent
            })
            print("\n" + "-" * 50)

        print("Completed all examples for this epoch.")

    return results

# Run the experiment
epochs = 3  # Number of epochs
results = adaptive_boundary_experiment_with_reverse(dataset, epochs)

# Analyze Results
correct_count = sum(1 for r in results if r["forward_similarity"] >= 0.9 and r["backward_consistency"])
print(f"Accuracy (Forward and Backward Consistent): {correct_count}/{len(results)} ({correct_count / len(results) * 100:.2f}%)")
