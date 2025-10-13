import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./shakespeare_sonnet_model")
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token exists
model = AutoModelForCausalLM.from_pretrained("./shakespeare_sonnet_model")

# Ensure model config matches tokenizer
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Function to generate a Shakespearean sonnet safely
def generate_sonnet(prompt):
    if not prompt.strip():
        return "Please enter a valid prompt."

    # Tokenize input with special tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    # Clip any token IDs that exceed vocab size
    inputs['input_ids'] = torch.clamp(inputs['input_ids'], max=model.config.vocab_size - 1)

    # Ensure max_length does not exceed model's context size
    max_len = min(200, model.config.n_positions - inputs['input_ids'].shape[1])

    # Generate text safely
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )

    # Decode and format
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sonnet_lines = lines[:14]  # Limit to 14 lines
    formatted_sonnet = "\n".join(sonnet_lines)

    return formatted_sonnet

# Gradio interface with bigger input/output boxes
iface = gr.Interface(
    fn=generate_sonnet,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Write your Shakespearean prompt here...",
            lines=5  # Bigger input box
        ),
    ],
    outputs=gr.Textbox(
        label="Generated Sonnet",
        lines=15  # Bigger output box
    ),
    title="Shakespeare Sonnet Generator",
    description="Enter a prompt and generate a Shakespearean-style 14-line sonnet safely."
)

if __name__ == "__main__":
    iface.launch()
