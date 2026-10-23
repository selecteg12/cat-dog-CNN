"""Пакетное распознавание папки или ZIP без повторного обучения."""

import argparse
import csv
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from .images import labels_from_probabilities, preprocess_image


def photo_id(filename):
    path = PurePosixPath(str(filename).replace("\\", "/"))
    if ("__MACOSX" in path.parts or any(part.startswith(".") for part in path.parts)
            or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}):
        return None
    return path.stem


@contextmanager
def image_source(input_path):
    """ZIP читается без распаковки; пиксели загружаются только для пакета."""
    input_path = Path(input_path)
    if input_path.is_dir():
        entries = [(photo_id(p.relative_to(input_path).as_posix()), p)
                   for p in input_path.rglob("*") if p.is_file()]
        entries = [(key, ref) for key, ref in entries if key is not None]
        yield validate_entries(entries), preprocess_image
    elif zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            entries = [(photo_id(info.filename), info) for info in archive.infolist()
                       if not info.is_dir()]
            entries = [(key, ref) for key, ref in entries if key is not None]

            def load(info):
                with archive.open(info) as source:
                    return preprocess_image(source)

            yield validate_entries(entries), load
    else:
        raise ValueError("Вход должен быть папкой с изображениями или ZIP-архивом")


def validate_entries(entries):
    if not entries:
        raise ValueError("Не найдены JPG/JPEG/PNG")
    ids = [key for key, _ in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Повторяются id: имена без расширения должны быть уникальны")
    return sorted(entries, key=lambda item: (
        0 if item[0].isdigit() else 1,
        int(item[0]) if item[0].isdigit() else item[0]))


def check_model(model):
    from tensorflow import keras

    if tuple(model.input_shape[1:]) != (128, 128, 3) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError("Модель должна принимать RGB 128×128 и иметь один выход")
    normalization = model.get_layer("normalize_0_1")
    head = model.get_layer("dog_probability")
    if (not isinstance(normalization, keras.layers.Rescaling)
            or not np.isclose(normalization.scale, 1 / 255)
            or normalization.offset != 0
            or model.layers[-1] is not head
            or head.get_config()["activation"] != "sigmoid"):
        raise ValueError("Нужна модель проекта: /255 внутри сети, выход dog_probability")


def predict_rows(model, entries, load, batch_size=32):
    if batch_size < 1:
        raise ValueError("Размер пакета должен быть положительным")
    rows = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start:start + batch_size]
        batch = []
        for key, ref in chunk:
            try:
                batch.append(load(ref))
            except Exception as error:
                raise ValueError(f"Не удалось прочитать фото id={key}") from error
        scores = np.asarray(model(np.stack(batch), training=False).numpy())
        if scores.shape != (len(chunk), 1):
            raise ValueError("Число или форма предсказаний не соответствует пакету")
        labels = labels_from_probabilities(scores[:, 0])
        rows.extend((key, int(label)) for (key, _), label in zip(chunk, labels))
        print(f"Обработано {len(rows)}/{len(entries)}")
    return rows


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def download_archive(file_id):
    import gdown

    if not file_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in file_id):
        raise ValueError("Некорректный Google Drive file id")
    cache = Path(".cache")
    cache.mkdir(exist_ok=True)
    destination = cache / f"{file_id}.zip"
    if not zipfile.is_zipfile(destination):
        temporary = destination.with_suffix(".zip.part")
        gdown.download(id=file_id, output=str(temporary), quiet=False)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("По ссылке не получен ZIP; проверьте доступ к файлу")
        temporary.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Папка или ZIP с фотографиями")
    source.add_argument("--drive-file-id", help="ID общедоступного ZIP на Google Drive")
    parser.add_argument("--model", type=Path, required=True, help="Свой best_model.keras")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size должен быть больше нуля")
        if not args.model.is_file():
            raise ValueError(f"Не найдена обученная модель: {args.model}")
        from tensorflow import keras

        model = keras.models.load_model(args.model, compile=False)
        check_model(model)
        input_path = download_archive(args.drive_file_id) if args.drive_file_id else args.input
        with image_source(input_path) as (entries, load):
            rows = predict_rows(model, entries, load, args.batch_size)
        write_csv(rows, args.output)
        print(f"Сохранено {len(rows)} ответов: {args.output}")
    except (ValueError, OSError, ImportError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
