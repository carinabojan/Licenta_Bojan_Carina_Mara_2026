import os
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PromptCNN(nn.Module):
    def __init__(self, k=1, dropout=0.20, freeze_backbone=True, num_pos=1, temperature=10):
        super().__init__()
        self.k = k
        self.num_pos = num_pos
        self.temperature = temperature
        self._frozen_modules = []

        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = backbone.conv1.weight.data.clone()
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_backbone:
            for m in [self.conv1, self.bn1, self.layer1, self.layer2,
                      self.layer3, self.layer4]:
                for p in m.parameters():
                    p.requires_grad = False

        self.heatmap_conv = nn.Conv2d(2048, k, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        for m in self._frozen_modules:
            m.eval()
        return self

    def forward(self, x):
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.layer1(x)
        x  = self.layer2(x)
        x  = self.layer3(x)
        f4 = self.layer4(x)

        B, _, H, W = f4.shape
        heatmap = self.heatmap_conv(f4)
        heatmap = heatmap.view(B, self.k, -1)
        heatmap = torch.softmax(heatmap * self.temperature, dim=-1)
        heatmap = heatmap.view(B, self.k, H, W)

        grid_x = torch.linspace(0, 1, W, device=f4.device).view(1, 1, 1, W).expand(B, self.k, H, W)
        grid_y = torch.linspace(0, 1, H, device=f4.device).view(1, 1, H, 1).expand(B, self.k, H, W)

        pred_x = (heatmap * grid_x).sum(dim=[2, 3])
        pred_y = (heatmap * grid_y).sum(dim=[2, 3])

        return torch.stack([pred_x, pred_y], dim=-1)


def load_prompt_cnn(checkpoint_path: str) -> tuple[PromptCNN, dict]:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg         = ckpt["config"]
        state_dict  = ckpt["model_state_dict"]
        num_points  = ckpt.get("num_points", cfg["num_pos_points"])
        point_labels = np.array(ckpt.get("point_labels",
                           [1] * cfg["num_pos_points"] + [0] * cfg["num_neg_points"]),
                           dtype=np.int32)
    else:
        cfg = {"num_pos_points": 1, "num_neg_points": 0,
               "dropout": 0.20, "temperature": 10, "image_size": 512}
        state_dict   = ckpt
        num_points   = 1
        point_labels = np.array([1], dtype=np.int32)

    model = PromptCNN(
        k           = num_points,
        dropout     = cfg["dropout"],
        temperature = cfg["temperature"],
        num_pos     = cfg["num_pos_points"],
        freeze_backbone=False,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).float().eval()

    meta = {
        "cfg":          cfg,
        "num_points":   num_points,
        "point_labels": point_labels,
        "image_size":   cfg.get("image_size", 512),
    }
    print(f"[model_utils] PromptCNN loaded — k={num_points}, "
          f"labels={point_labels}, image_size={meta['image_size']}")
    return model, meta


def load_sam():
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path  = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    sam_model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor(sam_model, device=device)

    print(f"[model_utils] SAM3 loaded on {device}.")
    return sam_model, processor


def preprocess_image(image_np: np.ndarray, image_size: int) -> torch.Tensor:
    from PIL import Image
    pil = Image.fromarray(image_np).convert("RGB").resize(
        (image_size, image_size), resample=Image.BILINEAR
    )
    arr = np.array(pil, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).float()
    t   = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(DEVICE)
