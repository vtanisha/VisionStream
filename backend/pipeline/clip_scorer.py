"""CLIP ViT-B/32 zero-shot frame scoring against user-supplied text prompts.

Zero-shot works because CLIP was trained to put images and their captions at the same
place in one shared 512-d space. So an unseen image and an unseen sentence are directly
comparable by cosine similarity - no classifier head, no training, no fixed label set.
Swapping the prompt list changes behaviour at runtime.
"""

import threading

import numpy as np
import open_clip
import torch
from PIL import Image

from ..config import Settings


class ClipScorer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)

        self._lock = threading.Lock()
        self._prompts: list[str] = []
        self._text_features: torch.Tensor | None = None
        self.set_prompts(settings.prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        """Text encoding is cached — it only depends on the prompts, not the frame.

        Re-encoding per frame would waste a full text forward pass every time.
        """
        clean = [p.strip() for p in prompts if p and p.strip()]
        with self._lock:
            self._prompts = clean
            if not clean:
                self._text_features = None
                return
            tokens = self.tokenizer(clean).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            self._text_features = feats

    @property
    def prompts(self) -> list[str]:
        with self._lock:
            return list(self._prompts)

    def score(self, frame_bgr: np.ndarray) -> list[tuple[str, float]]:
        """Returns [(prompt, softmax_score)] sorted high->low. Empty if no prompts."""
        with self._lock:
            text_features = self._text_features
            prompts = list(self._prompts)
        if text_features is None:
            return []

        rgb = Image.fromarray(frame_bgr[:, :, ::-1])  # cv2 is BGR, CLIP expects RGB
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # both sides L2-normalised, so the dot product IS cosine similarity.
            # 100.0 is CLIP's learned logit scale; softmax turns similarities into a
            # distribution ACROSS THE GIVEN PROMPTS - it is not a calibrated probability.
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        scores = probs[0].tolist()
        return sorted(zip(prompts, scores), key=lambda x: -x[1])

    def warmup(self) -> None:
        self.score(np.zeros((240, 320, 3), dtype=np.uint8))
