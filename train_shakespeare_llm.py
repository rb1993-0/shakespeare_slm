import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

DATA_FILE = "data/shakespeare_sonnets.txt"
MODEL_DIR = "shakespeare_sonnet_model"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_dataset(tokenizer, file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

def train():
    print("🚀 Starting training...")

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=256,
        n_ctx=256,
        n_embd=512,
        n_layer=8,
        n_head=8
    )

    model = GPT2LMHeadModel(config)

    dataset = get_dataset(tokenizer, DATA_FILE)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=75,  # ⬆️ Longer training — helps model grasp Shakespearean style
        per_device_train_batch_size=2,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=50,  # More frequent logs
        learning_rate=3e-5,  # 🔹 Add this for stability
        warmup_steps=100,  # 🔹 Gradually ramp up LR
        weight_decay=0.01,  # 🔹 Slight regularization to prevent overfitting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"✅ Training complete! Model saved at {MODEL_DIR}")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print("❌ Dataset not found! Run crawl_shakespeare_full.py first.")
    else:
        train()
