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

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate a Shakespearean sonnet safely
def generate_sonnet(prompt, temperature, top_k, top_p):
    if not prompt.strip():
        return "Please enter a valid prompt."

    # Tokenize input with special tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    ).to(device)

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
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=2.0,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and remove the prompt from generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Split into lines of ~10 words each
    words = text.split()
    lines = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
    sonnet_lines = lines[:14]  # Limit to 14 lines
    formatted_sonnet = "\n".join(sonnet_lines)

    return formatted_sonnet

# Gradio interface with sliders and tooltips
iface = gr.Interface(
    fn=generate_sonnet,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Write your Shakespearean prompt here...",
            lines=5
        ),
        gr.Slider(
            minimum=0.1, maximum=1.5, step=0.05, value=0.75,
            label="Temperature",
            info="Controls creativity: low=deterministic, high=creative/unpredictable"
        ),
        gr.Slider(
            minimum=1, maximum=100, step=1, value=40,
            label="Top-k",
            info="Limits next-word choices to top k probable tokens. Low=conservative, High=diverse"
        ),
        gr.Slider(
            minimum=0.1, maximum=1.0, step=0.01, value=0.9,
            label="Top-p (nucleus sampling)",
            info="Limits choices by cumulative probability. Low=focused, High=creative/diverse"
        ),
    ],
    outputs=gr.Textbox(
        label="Generated Sonnet",
        lines=15
    ),
    title="Shakespeare Sonnet Generator",
    description="Enter a prompt and adjust temperature, top_k, and top_p to control the style of your 14-line Shakespearean sonnet."
)

if __name__ == "__main__":
    iface.launch()
