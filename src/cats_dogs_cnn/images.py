"""Обработка фотографии совпадает с функцией в учебном ноутбуке."""

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (128, 128)
CLASS_TO_INDEX = {"cat": 0, "dog": 1}


def preprocess_image(source):
    """Путь или бинарный поток → RGB float32, (128, 128, 3), 0–255.

    Нормализация выполняется слоем Rescaling внутри модели.
    """
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Ожидается одно изображение, а не анимация")
        rgb = ImageOps.exif_transpose(image).convert("RGB").copy()
    resized = rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def labels_from_probabilities(probabilities):
    """Один sigmoid — оценка dog. Строго p > 0,5 означает собаку."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(values) & (values >= 0) & (values <= 1)):
        raise ValueError("Модель вернула некорректные оценки")
    return (values > 0.5).astype(np.int32)
