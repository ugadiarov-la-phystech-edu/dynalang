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
    
    def __init__(self, config_path, checkpoint_path, image_size, device, backbone_input_size=0,
                 token_mode='slots', pool_features='backbone', pool='soft', eps=1e-8):
        torch.nn.Module.__init__(self)
        self._device = torch.device(device)
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._image_size = image_size
        self._backbone_input_size = backbone_input_size
        assert token_mode in ('slots', 'object_centric')
        assert pool_features in ('backbone', 'features')
        assert pool in ('soft', 'hard')
        self._token_mode = token_mode
        self._pool_features = pool_features
        self._pool = pool
        self._eps = eps
        self._out_dim = None

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

        # Filter to only the modules used for extraction.
        prefixes = ('initializer.', 'encoder.', 'processor.')
        filtered_state_dict = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    filtered_state_dict[key] = value
                    break

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        assert len(missing_keys) == 0, f'Missing keys: {missing_keys}'
        assert len(unexpected_keys) == 0, f'Unexpected keys: {unexpected_keys}'

        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

        # In object-centric mode the output dim equals the pooled feature dim
        # (e.g. 384 for raw DINO features), which differs from slot_dim. Determine
        # it with a dummy forward so `dim` is correct before the first real call.
        if self._token_mode == 'object_centric':
            size = self.backbone_input_size or self._image_size
            with torch.no_grad():
                dummy = torch.zeros(1, 3, size, size, device=self._device)
                self._out_dim = self._forward_torch(dummy)[0].shape[-1]

    @property
    def n_slots(self):
        return self._model_config.initializer.n_slots

    @property
    def dim(self):
        if self._token_mode == 'object_centric':
            return self._out_dim
        return self._model_config.initializer.dim

    @property
    def carry_dim(self):
        return self._model_config.initializer.dim

    @property
    def backbone_input_size(self):
        return self._backbone_input_size if self._backbone_input_size else None

    def __call__(self, images, previous_slots=None):
        """
        Extract slots from images (numpy interface).
        
        Args:
            images: numpy array (B, H, W, C) uint8 [0, 255]
            previous_slots: numpy array (B, n_slots, carry_dim) or None. This is
                the recurrent carry state (the predictor's output from the
                previous call)
        
        Returns:
            slots: numpy array (B, n_slots, dim) float32, containing either
                corrected slots or pooled object-centric tokens
            predicted: numpy array (B, n_slots, carry_dim) float32 (predictor
                output, the correct value to pass back in as `previous_slots`)
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
        slots, predicted = self._forward_torch(batch_images, previous_slots)
        
        # Convert to numpy
        return slots.detach().cpu().numpy(), predicted.detach().cpu().numpy()

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

        # Object-centric tokens use slot attention only for masks, so initialize
        # independently for every frame instead of threading recurrent state.
        slots_initial = None if self._token_mode == 'object_centric' else previous_slots
        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        if self._input_type == "video":
            # We always feed one frame at a time (T=1), so ScanOverTime's own
            # internal step counter would always start at 0 and incorrectly
            # trigger `first_step_corrector_args` on every single call, not
            # just on genuine first frames of an episode. Call the wrapped
            # LatentProcessor directly with the *true* time step instead:
            # 0 only when there is no recurrent state to warm-start from
            # (i.e. a real first frame), non-zero for continuing frames.
            time_step = (
                0 if self._token_mode == 'object_centric' or previous_slots is None
                else 1
            )
            processor_output = self.processor.module(slots_initial, features[:, 0], time_step)
        else:
            processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]
        predicted = processor_output["state_predicted"]

        if self._token_mode == 'slots':
            return slots, predicted

        masks = processor_output["corrector"]["masks"]
        # Patch features to pool: raw DINO (backbone) or projected FrameEncoder feats.
        pool_feats = (
            encoder_output["backbone_features"]
            if self._pool_features == "backbone"
            else features
        )
        if self._input_type == "video":
            pool_feats = pool_feats[:, 0]

        return self._pool_tokens(masks, pool_feats), predicted

    def _pool_tokens(self, masks, features):
        """Pool patch features into one token per object mask.

        Args:
            masks: (B, n_slots, n_patches)
            features: (B, n_patches, dim)
        Returns:
            tokens: (B, n_slots, dim)
        """
        if self._pool == "hard":
            # Assign each patch to its argmax object, then average pool.
            assign = masks.argmax(dim=1)  # (B, n_patches)
            weights = F.one_hot(assign, num_classes=masks.shape[1]).float()
            weights = weights.permute(0, 2, 1)  # (B, n_slots, n_patches)
        else:
            # Soft (mask-weighted) average pooling.
            weights = masks
        denom = weights.sum(dim=-1, keepdim=True) + self._eps  # (B, n_slots, 1)
        tokens = torch.einsum("bsf, bfd -> bsd", weights, features) / denom
        return tokens
