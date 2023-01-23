from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import trange
import numpy as np


@dataclass
class QueryOutput:
    product_name: str
    url: str
    score: float


class ImageIndex:

    def __init__(
            self,
            keys: List[Path],
            images: List[Image.Image],
            processor: CLIPProcessor,
            tokenizer: CLIPTokenizerFast,
            model: torch.nn.Module,
            device: str,
    ):
        self.keys = keys
        self.images = images
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._build()

    def _build(self, batch_size: int = 16) -> None:
        processed = []
        for i in trange(0, len(self.images), batch_size):
            batch = self.images[i: i + batch_size]
            batch = self.processor(
                text=None,
                images=batch,
                return_tensors="pt",
                padding=True
            )["pixel_values"].to(self.device)
            batch_emb = self.model.get_image_features(pixel_values=batch).squeeze(0).cpu().detach().numpy()
            processed.append(batch_emb)
        self.index = np.concatenate(processed, axis=0)

    def query(self, text: str, n: int = 1):
        if n < 1:
            n = 1
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        text_emb = self.model.get_text_features(**inputs).cpu().detach()
        similarities = torch.nn.functional.cosine_similarity(text_emb, torch.tensor(self.index))
        return max_similarity_per_product(self.keys, similarities, n)


def read_images(path_pattern: str) -> Tuple[List[Path], List[Image.Image]]:
    path = Path()
    paths = path.glob(path_pattern)
    images = []
    keys = []
    for path in paths:
        try:
            images.append(Image.open(path))
            keys.append(path)
        # skip broken images
        except UnidentifiedImageError:
            print(f"Skipping image at {path}")
    return keys, images


def max_similarity_per_image(
        image_paths: List[Path], similarities: torch.tensor, n: int
) -> List[QueryOutput]:
    top = torch.argsort(similarities, descending=True)[:n]
    return [
        QueryOutput(
            product_name=image_paths[i].parts[-2],
            url=str(image_paths[i]),
            score=float(similarities[i])
        ) for i in top
    ]


def max_similarity_per_product(image_paths: List[Path], similarities: torch.tensor, n: int):
    seen = set()
    output = []
    top = torch.argsort(similarities, descending=True)
    for i in top:
        if len(output) >= n:
            break
        cur_product = image_paths[i].parts[-2]
        if cur_product not in seen:
            seen.add(cur_product)
            output.append(
                QueryOutput(
                    product_name=cur_product,
                    url=str(image_paths[i]),
                    score=float(similarities[i])
                )
            )
    return output


def default_init():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")
    model_name = "openai/clip-vit-base-patch16"

    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    keys, images = read_images("./no_bg/**/*.jpg")

    index = ImageIndex(keys, images, processor, tokenizer, model, device)

    return index


