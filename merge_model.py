import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the LoRA adapter.")
    parser.add_argument("--final_model_path", type=str, required=True, help="Path to save the new 16-bit merged model.")
    args = parser.parse_args()
    # 1. Load the model in 4-bit initially (same as training)
    base_model_path = args.base_model_path
    adapter_path = args.adapter_path
    final_model_path = args.final_model_path

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 2. Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # 3. Dequantize and Merge
    # This is the magic step. We merge the LoRA weights into the base weights.
    # Note: You usually have to reload the base model in 16-bit to merge cleanly,
    # but if you are VRAM constrained, you might need to merge on CPU.


    del model
    torch.cuda.empty_cache()

    # RELOAD base model in 16-bit (CPU offloading if necessary)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu" # Load to CPU RAM to avoid OOM
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() # Merges LoRA into base weights

    # 4. Save the new 16-bit model
    model.save_pretrained(final_model_path)

if __name__ == "__main__":
    main()