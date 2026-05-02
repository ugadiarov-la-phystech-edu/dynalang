import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf

from embodied.torch.ocr.slotcontrast import configuration, modules
from embodied.core.slot_extractor import SlotExtractor

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class SlotContrastExtractor(torch.nn.Module, SlotExtractor):
    """
    SlotContrast model wrapper for dynalang.
    
    Adapts SlotContrast to the SlotExtractor interface:
    - Accepts numpy images (H, W, C) uint8 [0, 255]
    - Returns numpy slots (n_slots, dim) float32
    """
    
    def __init__(self, config_path, checkpoint_path, image_size, device, backbone_input_size=0):
        torch.nn.Module.__init__(self)
        self._device = torch.device(device)
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._image_size = image_size
        self._backbone_input_size = backbone_input_size

        config = configuration.load_config(config_path)
        self._model_config = config.model

        self._normalization = torchvision.transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )

        # Build model components from slotcontrast config
        self.initializer = modules.build_initializer(self._model_config.initializer)
        self.encoder = modules.build_encoder(self._model_config.encoder, "FrameEncoder")

        grouper = modules.build_grouper(self._model_config.grouper)

        input_type = self._model_config.get("input_type", "image")
        if input_type == "image":
            self.processor = modules.LatentProcessor(grouper, predictor=None)
        elif input_type == "video":
            self.encoder = modules.MapOverTime(self.encoder)
            predictor = None
            if self._model_config.predictor is not None:
                from embodied.torch.ocr.slotcontrast.modules.utils import build_module
                predictor = build_module(self._model_config.predictor)
            if self._model_config.latent_processor:
                self.processor = modules.build_video(
                    self._model_config.latent_processor,
                    "LatentProcessor",
                    corrector=grouper,
                    predictor=predictor,
                )
            else:
                self.processor = modules.LatentProcessor(grouper, predictor)
            self.processor = modules.ScanOverTime(self.processor)
        else:
            raise ValueError(f"Unknown input type {input_type}")

        self._input_type = input_type

        # Load checkpoint weights
        state_dict = torch.load(
            self._checkpoint_path, weights_only=False, map_location='cpu'
        )
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter to only the modules we have (initializer, encoder, processor)
        filtered_state_dict = {}
        for key, value in state_dict.items():
            for prefix in ('initializer.', 'encoder.', 'processor.'):
                if key.startswith(prefix):
                    filtered_state_dict[key] = value
                    break

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        assert len(missing_keys) == 0, f'Missing keys: {missing_keys}'
        assert len(unexpected_keys) == 0, f'Unexpected keys: {unexpected_keys}'

        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_slots(self):
        return self._model_config.initializer.n_slots

    @property
    def dim(self):
        return self._model_config.initializer.dim

    @property
    def backbone_input_size(self):
        return self._backbone_input_size if self._backbone_input_size else None

    def __call__(self, images, previous_slots=None):
        """
        Extract slots from images (numpy interface).
        
        Args:
            images: numpy array (B, H, W, C) uint8 [0, 255]
            previous_slots: numpy array (B, n_slots, dim) or None
        
        Returns:
            slots: numpy array (B, n_slots, dim) float32
        """
        # Convert numpy to torch
        batch_images = torch.as_tensor(
            images.transpose(0, 3, 1, 2),  # (B, H, W, C) -> (B, C, H, W)
            dtype=torch.float32,
            device=self._device
        ) / 255.0
        
        # Resize if needed
        if self.backbone_input_size is not None:
            h, w = batch_images.shape[2], batch_images.shape[3]
            target = self.backbone_input_size
            if h != target or w != target:
                batch_images = F.interpolate(
                    batch_images, size=(target, target),
                    mode='bilinear', align_corners=False
                )
        
        # Convert previous_slots if provided
        if previous_slots is not None:
            previous_slots = torch.as_tensor(
                previous_slots, dtype=torch.float32, device=self._device
            )
        
        # Forward pass
        slots = self._forward_torch(batch_images, previous_slots)
        
        # Convert to numpy
        return slots.detach().cpu().numpy()

    def _forward_torch(self, image, previous_slots=None):
        """Internal forward pass with torch tensors."""
        encoder_input = self._normalization(image)
        batch_size = image.size()[0]

        # For video input_type, encoder/processor are wrapped in MapOverTime/ScanOverTime
        # which expect a time dimension (B, T, ...). Add a fake T=1 dimension.
        if self._input_type == "video":
            encoder_input = encoder_input.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = previous_slots
        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]

        # Remove the fake time dimension
        if self._input_type == "video":
            slots = slots[:, 0]  # (B, 1, n_slots, dim) -> (B, n_slots, dim)

        return slots
