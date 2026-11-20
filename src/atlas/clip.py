"""open_clip wrapper: image and text embeddings in a shared space (normalised)."""
import numpy as np


class ClipEmbedder:
    def __init__(self, name, pretrained):
        self.name, self.pretrained, self._m = name, pretrained, None

    def _load(self):
        if self._m is None:
            import open_clip

            model, _, pre = open_clip.create_model_and_transforms(
                self.name, pretrained=self.pretrained)
            self._m = (model.eval(), pre, open_clip.get_tokenizer(self.name))
        return self._m

    def encode_images(self, paths, batch=32):
        import torch
        from PIL import Image

        model, pre, _ = self._load()
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            with torch.no_grad():
                out.append(model.encode_image(x, normalize=True).numpy())
        return np.concatenate(out).astype("float32")

    def encode_text(self, texts):
        import torch

        model, _, tok = self._load()
        with torch.no_grad():
            return model.encode_text(tok(texts), normalize=True).numpy().astype("float32")
