from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch16"

tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)