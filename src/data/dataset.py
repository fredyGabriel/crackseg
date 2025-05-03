# import os # No longer needed
import random
import warnings
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import PIL.ImageOps
from pathlib import Path
import cv2

# Import the transform functions
from .transforms import (
    get_basic_transforms,
    apply_transforms,
    get_transforms_from_config
)

# Define SourceType at module level for type hinting
SourceType = Union[str, Path, PIL.Image.Image, np.ndarray]
# Define cache_item_type at module level
CacheItemType = Tuple[Optional[PIL.Image.Image], Optional[PIL.Image.Image]]


class CrackSegmentationDataset(Dataset):
    """
    PyTorch Dataset for crack segmentation.
    Loads image/mask pairs from a provided list or scans directories.
    Applies transformations using Albumentations.
    """

    def __init__(
        self,
        mode: str,  # Mode is required for transforms
        image_size: Optional[Tuple[int, int]] = None,
        # data_root: Optional[str] = None, # Keep for potential future use?
        samples_list: Optional[List[Tuple[str, str]]] = None,
        seed: Optional[int] = None,
        in_memory_cache: bool = False,
        config_transform: Optional[dict] = None,
        max_samples: Optional[int] = None  # Nuevo argumento opcional
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'. Determines transforms.
            image_size (tuple, optional): Target size (height, width) for
                resizing.
            samples_list (List[Tuple[str, str]], optional):
                Pre-defined list of (image_path, mask_path) tuples.
                If not provided, requires a different initialization
                method (e.g., via data_root, currently implies scanning,
                which is removed).
                If provided, these paths are used directly.
            seed (int, optional): Random seed for reproducibility.
            in_memory_cache (bool): If True, cache all data in RAM.
                Note: Cache stores raw PIL Images.
                Transforms applied after cache load.
            config_transform (dict, optional): Dict with transform config
                (Hydra YAML).
            max_samples (int, optional): If set and > 0, limits the number
                of samples loaded for this dataset (for fast testing).
        """
        if mode not in ["train", "val", "test"]:
            msg = f"Invalid mode: {mode}. Use 'train', 'val', or 'test'."
            raise ValueError(msg)
        self.mode = mode
        self.seed = seed
        self.in_memory_cache = in_memory_cache
        # self.data_root = data_root # Store if needed later?
        self.samples: List[Tuple[str, str]] = []

        if samples_list is not None:
            self.samples = samples_list
            if not self.samples:
                warnings.warn(
                    f"Provided samples_list for mode '{mode}' is empty."
                )
        # Removed the data_root scanning logic
        # elif data_root is not None:
        #     self._scan_directories() # Scan only if samples_list not given
        else:
            # If samples_list is None, we currently have no way to get data.
            # Raise error or expect data_root to be used by a subclass/factory?
            # For now, assume samples_list is the primary way.
            raise ValueError("samples_list must be provided.")

        # Limitar el número de muestras si se especifica max_samples
        if max_samples is not None and max_samples > 0:
            original_count = len(self.samples)
            # Asegurar que no intentamos tomar más muestras de las disponibles
            max_samples = min(max_samples, original_count)
            self.samples = self.samples[:max_samples]
            final_count = len(self.samples)
            print(
                f"DEBUG - Dataset '{mode}': Limitado de "
                f"{original_count} a {final_count} muestras"
            )
        else:
            print(
                f"DEBUG - Dataset '{mode}': Usando todas las "
                f"{len(self.samples)} muestras disponibles (sin límite)"
            )

        # Type hint for cache
        self._cache: Optional[List[CacheItemType]] = None

        # Selección de transformaciones: config dict > image_size > default
        if config_transform is not None:
            self.transforms = get_transforms_from_config(
                config_transform, self.mode
            )
        elif image_size is not None:
            self.transforms = get_basic_transforms(
                mode=self.mode, image_size=image_size
            )
        else:
            # Fallback: usar valores por defecto
            self.transforms = get_basic_transforms(mode=self.mode)

        # Build cache based on the final self.samples list
        if self.in_memory_cache and self.samples:
            self._build_cache()

        if self.seed is not None:
            self._set_seed()

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        if self.seed is None:  # Guard clause
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            warnings.warn("torch is not installed. Cannot set torch seed.")

    def _build_cache(self):
        """Cache all images and masks in memory as PIL Images."""
        self._cache = []
        for img_path, mask_path in self.samples:
            try:
                # Load as PIL for caching
                image = PIL.Image.open(img_path)
                image = PIL.ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                mask = PIL.Image.open(mask_path)
                mask = PIL.ImageOps.exif_transpose(mask)
                mask = mask.convert("L")  # Grayscale

                self._cache.append((image.copy(), mask.copy()))
                # Close files after copying
                image.close()
                mask.close()

            except Exception as e:
                warnings.warn(
                    f"Could not cache image/mask: "
                    f"{img_path}, {mask_path}: {e}"
                )
                self._cache.append((None, None))  # Placehold. for failed cache

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return the transformed image and mask pair at the given index.

        Loads image/mask from cache or disk.
        Applies the pre-defined transformation pipeline.
        Handles loading errors by attempting next sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'image' and 'mask'
                                     as torch tensors.

        Raises:
            RuntimeError: If no valid samples can be loaded after trying all.
        """
        attempts = 0
        max_attempts = len(self.samples)
        original_idx = idx

        while attempts < max_attempts:
            current_idx = (original_idx + attempts) % max_attempts
            # Define possible types for image/mask sources
            image_source: Optional[SourceType] = None
            mask_source: Optional[SourceType] = None

            # Try loading from cache first if enabled
            if self.in_memory_cache and self._cache is not None:
                cached_image, cached_mask = self._cache[current_idx]
                if cached_image is not None and cached_mask is not None:
                    # Convert PIL Image from cache to numpy array
                    image_source = np.array(cached_image)
                    mask_source = np.array(cached_mask)
                else:
                    # Cache entry failed or is missing, try next
                    attempts += 1
                    continue
            else:
                # Load paths from disk if not caching or cache failed
                image_source, mask_source = self.samples[current_idx]

            try:
                # Verificar explícitamente si las fuentes son rutas (strings)
                if isinstance(image_source, str) and isinstance(mask_source,
                                                                str):
                    # Cargar y transformar desde archivos
                    image = cv2.imread(image_source)
                    if image is None:
                        raise ValueError(f"Failed to load image: \
{image_source}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    mask = cv2.imread(mask_source, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Failed to load mask: {mask_source}")

                    # Asegurar que la máscara sea binaria (0/1)
                    mask = (mask > 127).astype(np.uint8)

                    # Aplicar transformaciones
                    if self.transforms is not None:
                        transformed = self.transforms(image=image, mask=mask)
                        # Ya debe ser un tensor
                        image_tensor = transformed['image']
                        mask_tensor = transformed['mask']

                        # Verificación adicional para asegurar que ambos sean
                        # tensores
                        if not isinstance(image_tensor, torch.Tensor):
                            image_tensor = torch.from_numpy(image_tensor)\
                                .permute(2, 0, 1).float() / 255.0
                        if not isinstance(mask_tensor, torch.Tensor):
                            mask_tensor = torch.from_numpy(mask_tensor)\
                                .unsqueeze(0).float()

                        # Asegurar que mask_tensor sea binario (0/1)
                        mask_tensor = (mask_tensor > 0.5).float()

                        return {"image": image_tensor, "mask": mask_tensor}
                    else:
                        # Sin transformaciones, convertir a tensor manualmente
                        image_tensor = torch.from_numpy(image)\
                            .permute(2, 0, 1).float() / 255.0
                        mask_tensor = torch.from_numpy(mask)\
                            .unsqueeze(0).float()
                        return {"image": image_tensor, "mask": mask_tensor}
                else:
                    # Apply transformations usando el código existente
                    # (no son strings)
                    sample = apply_transforms(
                        image=image_source,
                        mask=mask_source,
                        transforms=self.transforms
                    )

                    # Verificación final para asegurar que siempre devolvemos
                    # tensores
                    if not isinstance(sample['image'], torch.Tensor) or \
                       not isinstance(sample.get('mask', None), torch.Tensor):
                        raise TypeError(
                            "apply_transforms no retornó tensores PyTorch. "
                            "Revisar implementación."
                        )

                    return sample

            except Exception as e:
                img_path, mask_path = self.samples[current_idx]
                warnings.warn(
                    f"Error processing sample at index {current_idx} "
                    f"({img_path}, {mask_path}): {e}. Skipping."
                )
                attempts += 1
                # Optional: Invalidate cache entry?
                # if self.in_memory_cache and self._cache is not None:
                #     self._cache[current_idx] = (None, None)

        raise RuntimeError(
            "No valid image/mask pairs could be loaded/processed from dataset."
        )
