from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file
import requests
import os
from typing import Any


class Predictor(BasePredictor):
    def download_model(self, url: str, cache_dir: str = "models_cache") -> str:
        """
        Pobiera model z podanego URL do lokalnego katalogu cache.
        """
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, url.split("/")[-1])
        if not os.path.exists(filepath):
            print(f"Pobieranie modelu z {url}...")
            with open(filepath, "wb") as f:
                f.write(requests.get(url).content)
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
        image: Path = Input(
            description="Ścieżka do obrazu wejściowego",
        ),
        mask: Path = Input(
            description="Ścieżka do maski obrazu",
        ),
        num_inference_steps: int = Input(
            description="Liczba kroków generacji", 
            default=30,
        ),
        guidance_scale: float = Input(
            description="Stopień precyzji podpowiedzi", 
            default=7.5,
        ),
    ) -> Path:
        """
        Wykonuje inpainting przy użyciu dynamicznie pobranego modelu LoRA.
        Zwraca wygenerowany obraz.
        """
        # Pobranie modelu LoRA
        lora_path = self.download_model(model_url)

        # Załaduj bazowy model inpainting
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-inpainting"
        )
        pipeline.unet.load_attn_procs(load_file(lora_path))

        # Wykonaj inpainting
        result = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Zapisz wynik
        output_path = "output.png"
        result.images[0].save(output_path)
        return Path(output_path)
