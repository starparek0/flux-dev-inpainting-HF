# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input
import requests
import os
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        """
        Setup model initialization (if needed).
        """
        print("Model setup complete!")

    def download_model(self, model_url: str, cache_dir="models_cache") -> str:
        """
        Download model from a URL to a local cache directory.
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        filename = model_url.split("/")[-1]
        filepath = os.path.join(cache_dir, filename)

        # Download the file only if it doesn't already exist
        if not os.path.exists(filepath):
            print(f"Downloading model from {model_url}...")
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
            else:
                raise ValueError(f"Error downloading model: {response.status_code}")
        return filepath

    def predict(
        self,
        model_url: str = Input(
            description="Link do modelu LoRA na Hugging Face (.safetensors)",
            default="https://huggingface.co/user/model-name/resolve/main/model.safetensors",
        ),
        prompt: str = Input(
            description="Opis obrazu do wygenerowania",
            default="A beautiful landscape",
        ),
        image: str = Input(
            description="Link do obrazu bazowego",
        ),
        mask: str = Input(
            description="Link do maski obrazu",
        ),
        num_inference_steps: int = Input(
            description="Liczba kroków (im większa, tym lepsza jakość)",
            default=30,
        ),
        guidance_scale: float = Input(
            description="Stopień precyzji podpowiedzi",
            default=7.5,
        ),
    ) -> dict:
        """
        Perform inpainting with the LoRA model and input image.
        """
        # Download the LoRA model from the provided URL
        lora_path = self.download_model(model_url)

        # Load the base inpainting model
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-inpainting"
        )
        pipeline.unet.load_attn_procs(load_file(lora_path))

        # Perform inpainting
        result = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Save and return the output
        output_path = "output.png"
        result.images[0].save(output_path)
        return {"output": output_path}
