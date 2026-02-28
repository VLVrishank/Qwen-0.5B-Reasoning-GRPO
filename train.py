import torch
import re
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You must respond in this exact format: (you should use those tags like <think>,</think>,<answer>,</answer>)
<think>
your thoughts Step by step reasoning here to solve the problem
</think>
<answer>
Your Final answer here for the user
</answer>"""

def get_gsm8k_questions(split="train"):
    """
    Load GSM8K dataset and format prompts.

    Args:
        split: Dataset split to load (default: "train")

    Returns:
        Formatted dataset with prompts and answers
    """
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answers': x['answer'].split('####')[-1].strip()
    })
    return data

# Initialize dataset with 100 samples
dataset = get_gsm8k_questions().select(range(100))

# ==========================================
# 2. Reward Function with Logging
# ==========================================

# Global tracking variables
step_counter = 0
reward_history = []

def extract_answer(text):
    """
    Extract the core answer (number or key content) from text.

    Args:
        text: Input text string

    Returns:
        Cleaned answer string
    """
    # Convert to lowercase
    text = text.lower()

    # Remove common prefixes
    text = re.sub(r'\b(the answer is|the final answer is|equals|is|are)\b', '', text)

    # Remove currency symbols and units
    text = re.sub(r'[$£€¥]', '', text)

    # Extract number if present
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[0].strip()

    # Otherwise return cleaned text
    return text.strip()

def format_and_correctness_reward(prompts, completions, answers, **kwargs):
    """
    Reward function evaluating both format compliance and answer correctness.

    Format scoring (0-13 points):
        - <think> tag: 2 points
        - </think> tag: 2 points
        - <answer> tag: 2 points
        - </answer> tag: 2 points
        - Complete format bonus: 5 points

    Correctness scoring (0-10 points):
        - Correct answer: 10 points

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answers: List of ground truth answers
        **kwargs: Additional arguments

    Returns:
        List of reward scores
    """
    global step_counter, reward_history

    responses = [c[0]["content"] for c in completions]
    rewards = []
    format_scores = []
    correctness_scores = []

    for i, (response, ground_truth) in enumerate(zip(responses, answers)):
        format_score = 0.0
        correctness_score = 0.0

        # Log first response every 10 steps for inspection
        if i == 0 and len(rewards) % 10 == 0:
            print(f"\n--- Sample Response (Step {step_counter}) ---")
            print(response[:300])
            print("---\n")

        # Evaluate format compliance
        has_think_open = "<think>" in response.lower()
        has_think_close = "</think>" in response.lower()
        has_answer_open = "<answer>" in response.lower()
        has_answer_close = "</answer>" in response.lower()

        # Award points for each format element
        if has_think_open:
            format_score += 2.0
        if has_think_close:
            format_score += 2.0
        if has_answer_open:
            format_score += 2.0
        if has_answer_close:
            format_score += 2.0

        # Bonus for complete format
        if has_think_open and has_think_close and has_answer_open and has_answer_close:
            format_score += 5.0

            # Evaluate correctness only if format is complete
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()

                # Clean and compare answers
                extracted_clean = extract_answer(extracted)
                ground_truth_clean = extract_answer(ground_truth)

                if extracted_clean == ground_truth_clean:
                    correctness_score = 10.0
                    print(f"   CORRECT: '{extracted}' -> '{extracted_clean}' == '{ground_truth_clean}'")
                else:
                    print(f"   INCORRECT: '{extracted}' -> '{extracted_clean}' != '{ground_truth_clean}'")

        # Calculate total score
        total_score = format_score + correctness_score
        rewards.append(total_score)
        format_scores.append(format_score)
        correctness_scores.append(correctness_score)

    # Calculate batch statistics
    avg_reward = sum(rewards) / len(rewards)
    avg_format = sum(format_scores) / len(format_scores)
    avg_correctness = sum(correctness_scores) / len(correctness_scores)

    # Track history
    reward_history.append({
        'step': step_counter,
        'avg_reward': avg_reward,
        'avg_format': avg_format,
        'avg_correctness': avg_correctness,
        'max_reward': max(rewards),
        'min_reward': min(rewards)
    })

    # Log reward statistics
    print(f"\nSTEP {step_counter} REWARD SUMMARY:")
    print(f"   Average Total Reward: {avg_reward:.2f}")
    print(f"   - Format Score:       {avg_format:.2f} / 13.0")
    print(f"   - Correctness Score:  {avg_correctness:.2f} / 10.0")
    print(f"   Reward Range: [{min(rewards):.1f}, {max(rewards):.1f}]")

    step_counter += 1

    return rewards

# ==========================================
# 3. Training Configuration
# ==========================================

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Configure GRPO training parameters
training_args = GRPOConfig(
    output_dir="Qwen-CoT-Aggressive",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    generation_batch_size=8,
    max_completion_length=512,
    max_steps=150,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    report_to="none",
    generation_kwargs={
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
    }
)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=MODEL_ID,
    processing_class=tokenizer,
    reward_funcs=[format_and_correctness_reward],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

# ==========================================
# 4. Execute Training
# ==========================================
print("=" * 60)
print("GRPO TRAINING WITH REWARD TRACKING")
print("=" * 60)
print("Tracking metrics:")
print("  - Reward statistics per step")
print("  - Format scores (0-13 points)")
print("  - Correctness scores (0-10 points)")
print("=" * 60)

trainer.train()

# ==========================================
# 5. Post-Training Analysis
# ==========================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 60)

if reward_history:
    # Print reward progression
    print(f"\nReward Progression:")
    print(f"   Initial Average Reward: {reward_history[0]['avg_reward']:.2f}")
    print(f"   Final Average Reward:   {reward_history[-1]['avg_reward']:.2f}")
    print(f"   Total Improvement:      +{reward_history[-1]['avg_reward'] - reward_history[0]['avg_reward']:.2f}")

    # Print format learning
    print(f"\nFormat Compliance Learning:")
    print(f"   Initial: {reward_history[0]['avg_format']:.2f} / 13.0")
    print(f"   Final:   {reward_history[-1]['avg_format']:.2f} / 13.0")

    # Print correctness learning
    print(f"\nAnswer Correctness Learning:")
    print(f"   Initial: {reward_history[0]['avg_correctness']:.2f} / 10.0")
    print(f"   Final:   {reward_history[-1]['avg_correctness']:.2f} / 10.0")

    # Generate reward visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        steps = [r['step'] for r in reward_history]
        avg_rewards = [r['avg_reward'] for r in reward_history]
        format_scores = [r['avg_format'] for r in reward_history]
        correctness_scores = [r['avg_correctness'] for r in reward_history]

        # Create three-panel plot
        plt.figure(figsize=(12, 4))

        # Total reward plot
        plt.subplot(1, 3, 1)
        plt.plot(steps, avg_rewards, 'b-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title('Total Reward Over Time')
        plt.grid(True, alpha=0.3)

        # Format score plot
        plt.subplot(1, 3, 2)
        plt.plot(steps, format_scores, 'g-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Format Score')
        plt.title('Format Score (0-13)')
        plt.grid(True, alpha=0.3)

        # Correctness score plot
        plt.subplot(1, 3, 3)
        plt.plot(steps, correctness_scores, 'r-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Correctness Score')
        plt.title('Correctness Score (0-10)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('reward_progression.png', dpi=150, bbox_inches='tight')
        print(f"\nReward progression plot saved: reward_progression.png")

    except ImportError:
        print("\nNote: Install matplotlib to generate reward plots (pip install matplotlib)")

# Save final model
trainer.save_model("Qwen-CoT/final")
print("\nModel saved successfully to: Qwen-CoT/final")
