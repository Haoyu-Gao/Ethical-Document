---
{}
---
Model trained on the TinyStories Dataset, see https://arxiv.org/abs/2305.07759

------ EXAMPLE USAGE ---

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate completion
output = model.generate(input_ids, max_length = 1000, num_beams=1)

# Decode the completion
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(output_text)