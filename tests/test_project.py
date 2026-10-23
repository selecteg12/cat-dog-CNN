import ast
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from cats_dogs_cnn.images import preprocess_image, labels_from_probabilities
from cats_dogs_cnn.predict import image_source, predict_rows, write_csv

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebooks/cats_dogs_cnn.ipynb').read_text(encoding='utf-8'))


def notebook_functions(*names):
    """Проверяем настоящие функции ноутбука без запуска Drive, графиков и fit."""
    namespace = dict(np=np, pd=pd, Image=Image, ImageOps=ImageOps, Path=Path,
                     PurePosixPath=__import__('pathlib').PurePosixPath,
                     zipfile=zipfile, shutil=__import__('shutil'), json=json,
                     SEED=42, IMG_SIZE=(128, 128))
    nodes = []
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook>', 'exec'), namespace)
    return namespace


def image_bytes(color=128):
    stream = io.BytesIO()
    Image.new('L', (20, 10), color).save(stream, format='PNG')
    return stream.getvalue()


class FakeOutput:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class FakeModel:
    def __call__(self, batch, training):
        if training is not False:
            raise AssertionError('Inference должен отключать training')
        return FakeOutput((batch[:, 0, 0, 0] / 255).reshape(-1, 1))


class ProjectTests(unittest.TestCase):
    def test_notebook_clean_and_python_syntax(self):
        self.assertEqual(NOTEBOOK['nbformat'], 4)
        ids = [cell['id'] for cell in NOTEBOOK['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_preprocess_matches_notebook_and_keeps_255_range(self):
        for color in (0, 128, 255):
            data = image_bytes(color)
            prepared = preprocess_image(io.BytesIO(data))
            # Проверяем отдельно обработку обучения и автономной демонстрации.
            for index in (4, 34):
                functions = dict(np=np, Image=Image, ImageOps=ImageOps, IMG_SIZE=(128, 128))
                nodes = [node for node in ast.parse(''.join(NOTEBOOK['cells'][index]['source'])).body
                         if isinstance(node, ast.FunctionDef)
                         and node.name in {'read_rgb', 'preprocess_image'}]
                exec(compile(ast.Module(body=nodes, type_ignores=[]), '<preprocess>', 'exec'), functions)
                np.testing.assert_array_equal(prepared, functions['preprocess_image'](io.BytesIO(data)))
            self.assertEqual(prepared.shape, (128, 128, 3))
            self.assertEqual(prepared.dtype, np.float32)
            self.assertEqual(float(prepared.max()), color)

    def test_labels_threshold_and_invalid_values(self):
        np.testing.assert_array_equal(labels_from_probabilities([0, .4999, .5, .5001, 1]),
                                      [0, 0, 0, 1, 1])
        for value in (np.nan, np.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                labels_from_probabilities([value])

    def test_zip_predictions_preserve_ids_and_csv_has_two_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder) / 'images.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                for name, color in [('nested/10.png', 255), ('2.jpg', 0), ('1.png', 128)]:
                    archive.writestr(name, image_bytes(color))
                archive.writestr('__MACOSX/._1.png', b'ignored')
                archive.writestr('notes.txt', 'ignored')
            with image_source(archive_path) as (entries, load):
                rows = predict_rows(FakeModel(), entries, load, batch_size=2)
            self.assertEqual(rows, [('1', 1), ('2', 0), ('10', 1)])
            output = Path(folder) / 'result.csv'
            write_csv(rows, output)
            with output.open(newline='', encoding='utf-8') as source:
                csv_rows = list(csv.reader(source))
            self.assertEqual(csv_rows[0], ['id', 'label'])
            self.assertTrue(all(len(row) == 2 for row in csv_rows))
            self.assertEqual(len(csv_rows), 4)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for sub in ('a', 'b'):
                (Path(folder) / sub).mkdir()
                (Path(folder) / sub / '1.png').write_bytes(image_bytes())
            with self.assertRaisesRegex(ValueError, 'Повторяются id'):
                with image_source(folder):
                    pass

    def test_broken_image_stops_batch(self):
        with self.assertRaisesRegex(ValueError, 'id=broken'):
            predict_rows(FakeModel(), [('broken', io.BytesIO(b'not an image'))], preprocess_image)

    def test_deduplication_excludes_conflicting_groups(self):
        remove = notebook_functions('remove_duplicates')['remove_duplicates']
        frame = pd.DataFrame({'pixel_sha256': ['cat', 'cat', 'dog', 'conflict', 'conflict'],
                              'label': [0, 0, 1, 0, 1]})
        clean, duplicates, conflicts = remove(frame)
        self.assertEqual(set(clean.pixel_sha256), {'cat', 'dog'})
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(conflicts), 2)

    def test_stratified_split_is_balanced_disjoint_and_repeatable(self):
        split = notebook_functions('stratified_split')['stratified_split']
        frame = pd.DataFrame({'label': [0] * 20 + [1] * 20,
                              'pixel_sha256': [str(i) for i in range(40)]})
        result = split(frame)
        self.assertEqual([len(part) for part in result], [28, 6, 6])
        for part, repeated in zip(result, split(frame)):
            self.assertEqual(part.label.value_counts().nunique(), 1)
            pd.testing.assert_frame_equal(part, repeated)
        hashes = [set(part.pixel_sha256) for part in result]
        self.assertTrue(hashes[0].isdisjoint(hashes[1]))
        self.assertTrue(hashes[0].isdisjoint(hashes[2]))
        self.assertTrue(hashes[1].isdisjoint(hashes[2]))

    def test_zip_extraction_rejects_traversal_before_writing(self):
        extract = notebook_functions('safe_extract_zip')['safe_extract_zip']
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder) / 'bad.zip'
            destination = Path(folder) / 'out'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('normal.txt', 'must not be written')
                archive.writestr('../escape.txt', 'invalid')
            with self.assertRaises(ValueError):
                extract(archive_path, destination)
            self.assertFalse((destination / 'normal.txt').exists())
            self.assertFalse((Path(folder) / 'escape.txt').exists())

    def test_cache_invalidated_when_dataset_or_settings_change(self):
        compatible = notebook_functions('cache_is_compatible')['cache_is_compatible']
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'run.json'
            expected = {'dataset_sha256': 'first', 'seed': 42}
            self.assertFalse(compatible(config, expected))
            config.write_text(json.dumps(expected), encoding='utf-8')
            self.assertTrue(compatible(config, expected))
            self.assertFalse(compatible(config, {**expected, 'dataset_sha256': 'other'}))
            self.assertFalse(compatible(config, {**expected, 'seed': 7}))

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow отсутствует локально')
    def test_original_cnn_contract_and_forward_pass(self):
        from tensorflow import keras
        from cats_dogs_cnn.predict import check_model
        namespace = dict(keras=keras, SEED=42)
        cell = NOTEBOOK['cells'][20]
        definitions = [node for node in ast.parse(''.join(cell['source'])).body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id in {'augmentation', 'model'}
                           for target in node.targets)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<cnn>', 'exec'), namespace)
        model = namespace['model']
        check_model(model)
        self.assertEqual(sum(int(np.prod(weight.shape)) for weight in model.trainable_weights), 765249)
        output = model(np.zeros((2, 128, 128, 3), dtype=np.float32), training=False).numpy()
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(output) & (output >= 0) & (output <= 1)))


if __name__ == '__main__':
    unittest.main()
