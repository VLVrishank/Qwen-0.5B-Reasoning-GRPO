# Qwen-0.5B-Reasoning-GRPO
# Mini-R1: Inducing Reasoning in 0.5B Models via GRPO

**Can a 500M parameter model actually "think"?** This project explores the lower bounds of emergent reasoning by applying **Group Relative Policy Optimization (GRPO)** to Qwen-0.5B. Instead of traditional SFT, I use a zero-critic RL loop to force the model to develop Chain-of-Thought (CoT) capabilities on a minimal compute budget.

---

## The Hook: Why 0.5B?
Most people think reasoning is a function of scale. I wanted to prove it is a function of the **training objective**. By using GRPO, we eliminate the need for a separate Reward/Value model, allowing for extreme compute efficiency. Training a 0.5B model to follow a strict `<think>` and `<answer>` format while solving math is "Hard Mode" for alignment, and it provides immediate signal on reward function quality.

## Training Results
The reward progression graphs reveal a distinct two-phase learning process over the ~75 recorded steps:

1. **Format Mastery (Steps 0-15):** The model rapidly learns the required XML tag structure. The Format Score climbs aggressively and pegs at the maximum (13 points) very early in the training run.
2. **Logic Emergence (Steps 15-75):** Once the format is locked in, the Correctness Score begins to register. While highly volatile (bouncing between 0 and 8+ points), this confirms the model stops zeroing out and begins successfully exploring the mathematical logic needed to solve the prompts, which drives the Total Reward variance.

![Reward Progression](<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/e108bc86-e0c4-4918-8a65-13757c1ba537" />
)

---

## Logic Emergence (Example)
Here is an example of the 0.5B model solving a GSM8K-style problem after the initial GRPO steps:

> **User:** If Molly has 2 apples and buys 3 more, then gives 1 to Pete, how many does she have?
>
> **Model:**
> <think>
> 1. Start with 2 apples.
> 2. Add 3 apples: 2 + 3 = 5.
> 3. Give 1 apple away: 5 - 1 = 4.
> 4. The final count is 4.
> </think>
> <answer>
> 4
> </answer>

---

## Tech Stack & Methodology
* **Model:** Qwen2.5-0.5B-Instruct
* **Algorithm:** GRPO (via `trl`)
* **LoRA:** Rank 16 / Alpha 32
* **Reward Strategy:** * **Format (13 pts):** Strict XML tag compliance.
    * **Correctness (10 pts):** Deterministic regex-based answer verification.
* **First Principles:** Solves the RL "Cold Start" problem by rewarding format compliance first, which acts as a bridge to mathematical correctness.

---

## How to Run

```bash
pip install trl peft transformers datasets
python train.py
