"""Frozen SOLD SAVi encoder with explicit predictor carry between frames."""

import importlib
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from embodied.core.slot_extractor import SlotExtractor


class _UnusedScheduler:
    """Deserialization target for SOLD's unused training scheduler metadata."""


class _SoldUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('modeling.autoencoder.'):
            module = 'embodied.torch.ocr.savi._sold.' + module
        elif (module, name) == (
                'pl_bolts.optimizers.lr_scheduler',
                'LinearWarmupCosineAnnealingLR'):
            return _UnusedScheduler
        return super().find_class(module, name)


class _SoldPickle:
    # Lightning checkpoints contain the pickled autoencoder architecture.
    # Remap its classes locally without importing SOLD, Lightning or pl_bolts.
    Unpickler = _SoldUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    __name__ = 'sold_pickle'


class SAViExtractor(torch.nn.Module, SlotExtractor):
    def __init__(self, config_path, checkpoint_path, image_size, device,
                 backbone_input_size=0):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'SAVi checkpoint not found: {path}')
        with path.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/'):
                raise ValueError(
                    f'{path} is a Git LFS pointer, not weights. Run '
                    'scripts/fetch_savi_checkpoint.py or git lfs pull in SOLD.')
        # Only load trusted SOLD checkpoints: these include Python metadata.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False,
                                pickle_module=_SoldPickle)
        self.model = checkpoint['hyper_parameters']['autoencoder']
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError('SOLD checkpoint has no autoencoder architecture')
        predictor_type = importlib.import_module(
            'embodied.torch.ocr.savi._sold.modeling.autoencoder.savi.predictor'
        ).TransformerPredictor
        if type(self.model.predictor) is not predictor_type:
            raise ValueError('Only action-independent SOLD TransformerPredictor '
                             'checkpoints are supported by the slot interface')
        weights = {key.removeprefix('autoencoder.'): value
                   for key, value in checkpoint['state_dict'].items()
                   if key.startswith('autoencoder.')}
        self.model.load_state_dict(weights, strict=True)
        self._device = torch.device(device)
        self._image_size = tuple(self.model.decoder.image_size)
        if backbone_input_size and self._image_size != (backbone_input_size,) * 2:
            raise ValueError('backbone_input_size must match the SAVi checkpoint')
        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_slots(self):
        return self.model.num_slots

    @property
    def dim(self):
        return self.model.slot_dim

    @property
    def backbone_input_size(self):
        return self._image_size[0]

    @property
    def decoder(self):
        return self.model.decoder

    def _images(self, images):
        images = torch.as_tensor(np.asarray(images) if not torch.is_tensor(images)
                                 else images, device=self._device)
        if images.ndim != 4:
            raise ValueError('Expected a batch of RGB frames (B,H,W,3) or (B,3,H,W)')
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] != 3:
            raise ValueError('SAVi expects three RGB channels')
        uint8 = images.dtype == torch.uint8
        images = images.float()
        if uint8:
            images = images / 255.0
        elif not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('Float RGB frames must be finite and in [0,1]')
        if images.shape[-2:] != self._image_size:
            images = F.interpolate(images, size=self._image_size,
                                   mode='bilinear', align_corners=False)
        return images

    @torch.no_grad()
    def forward(self, images, previous_slots=None):
        images = self._images(images)
        batch = images.shape[0]
        if previous_slots is None:
            predicted = self.model.initializer(batch)
        else:
            predicted = torch.as_tensor(previous_slots, dtype=torch.float32,
                                        device=self._device)
            if predicted.shape != (batch, self.n_slots, self.dim):
                raise ValueError(f'Invalid SAVi carry shape: {tuple(predicted.shape)}')
        features = self.model.encoder(images)
        slots = self.model.corrector(features, predicted,
                                     is_first=previous_slots is None)
        # The next call consumes this prediction directly, exactly once.
        carry = self.model.predictor(slots, None)
        return slots, carry

    @torch.no_grad()
    def decode_slots(self, slots):
        """Decode (...,K,D) slots to RGB reconstructions and soft masks."""
        slots = torch.as_tensor(slots, dtype=torch.float32, device=self._device)
        leading = slots.shape[:-2]
        if tuple(slots.shape[-2:]) != (self.n_slots, self.dim):
            raise ValueError('Invalid SAVi slot shape')
        # SAVi spatially broadcasts every slot to 64x64. Decoding a long
        # training report at once would create a large GPU activation tensor.
        chunks = []
        flat = slots.reshape(-1, self.n_slots, self.dim)
        for chunk in flat.split(4):
            rgbs, masks = self.model.decoder(chunk)
            reconstruction = (rgbs * masks).sum(1).clamp(0, 1)
            chunks.append({key: value.cpu().numpy() for key, value in {
                'reconstructions': reconstruction, 'rgbs': rgbs, 'masks': masks}.items()})
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks])
                  for key in chunks[0]}
        return {key: value.reshape(*leading, *value.shape[1:])
                for key, value in arrays.items()}

    def decode_masks(self, slots):
        """(T,K,D) -> (T,K,H*W), for embodied's slot report renderer."""
        masks = self.decode_slots(slots)['masks']
        return masks.reshape(*masks.shape[:-3], int(np.prod(self._image_size)))
