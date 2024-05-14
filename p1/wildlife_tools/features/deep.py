import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features.base import FeatureExtractor
from wildlife_tools.tools import realize


class DeepFeatures(FeatureExtractor):
    """
    Extracts features using forward pass of pytorch model.

    Args:
        model: Pytorch model used for the feature extraction.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.

    """

    def __init__(
        self,
        model,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model

    def __call__(self, dataset: WildlifeDataset):
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        outputs = []
        for image, label in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(image.to(self.device))
                outputs.append(output.cpu())
        return torch.cat(outputs).numpy()

    @classmethod
    def from_config(cls, config):
        model = realize(config.pop("model"))
        return cls(model=model, **config)


class ClipFeatures:
    """
    Extract features using CLIP model (https://arxiv.org/pdf/2103.00020.pdf).
    Uses raw images of input WildlifeDataset (i.e. dataset.transform = None)

    Args:
        model: transformer.CLIPModel. Uses VIT-L backbone by default.
        processor: transformer.CLIPProcessor. Uses VIT-L processor by default.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        batch_size=128,
        num_workers=1,
        device="cpu",
    ):
        if model is None:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).vision_model

        if processor is None:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.transform = lambda x: processor(images=x, return_tensors="pt")[
            "pixel_values"
        ]

    def __call__(self, dataset: WildlifeDataset):
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        dataset.transforms = None  # Reset transforms.

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        outputs = []
        for image in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(self.transform(image).to(self.device)).pooler_output
                outputs.append(output.cpu())
        return torch.cat(outputs).numpy()
    

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

class DeepFeaturesWithTripletLoss(DeepFeatures):
    """
    Extracts features using forward pass of pytorch model with triplet loss.

    Args:
        model: Pytorch model used for the feature extraction.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.

    """

    def __init__(
        self,
        model,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        margin: float = 1.0
    ):
        super().__init__(model, batch_size, num_workers, device)
        self.margin = margin

    def triplet_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    def __call__(self, dataset: WildlifeDataset):
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        outputs = []
        for image, label in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(image.to(self.device))
                outputs.append(output.cpu())

        outputs = torch.cat(outputs)
        num_samples = len(outputs)
        features = outputs.view(num_samples // 3, 3, -1)  # Assuming triplet batches
        loss = 0
        for triplet in features:
            anchor, positive, negative = triplet
            loss += self.triplet_loss(anchor, positive, negative)

        loss /= len(features)
        return outputs.numpy(), loss.item()

