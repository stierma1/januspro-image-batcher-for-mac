import json
import os
import argparse
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import VLChatProcessor

import numpy as np
# import spaces  # Import spaces for ZeroGPU compatibility


# Load model and processor
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

def load_json_prompts(json_path):
    """Load JSON file containing array of prompt objects."""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 3,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


@torch.inference_mode()
# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   temperature=1.0,
                   guidance=9,
                   number_of_samples = 3,
                   model=None
                   ):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    t2i_temperature = temperature

    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=number_of_samples,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=number_of_samples)

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(number_of_samples)]

def save_images(images, filename_template, output_folder):
    """Save generated images to specified folder with formatted filenames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, img in enumerate(images, 1):
        # Generate unique filename
        filename = filename_template.replace("{#}", str(i))
        filepath = os.path.join(output_folder, filename)

        if isinstance(img, Image.Image):
            img.save(filepath)
        else:
            # Handle if images are file paths or other types
            pass

#Example of how to invoke
#python main.py --config ./sample-config.json --output ./sample-output
def main():
    parser = argparse.ArgumentParser(description='Process configuration file')
    parser.add_argument('--config', type=str, required=True,
                    help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, required=True,
                    help='Output directory path')
    
    args = parser.parse_args()
    # Verify config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist")
    prompts = load_json_prompts(args.config)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    output_folder = args.output

    for item in prompts:
        prompt = item['prompt']
        filename_template = item['fileNameTemplate']
        seed = item.get('seed', 1234)
        temperature = item.get('temperature', 1.0)
        guidance = item.get('guidance', 9)
        number_of_samples = item.get("number_of_samples", 3)

        # Generate images
        try:
            images = generate_image(prompt, seed, temperature, guidance, number_of_samples)

            # Save images to output folder
            save_images(images, filename_template, output_folder)

            print(f"Successfully processed prompt: {prompt}")
        except Exception as e:
            print(f"Error processing prompt: {prompt}. Error: {str(e)}")

if __name__ == "__main__":
    main()