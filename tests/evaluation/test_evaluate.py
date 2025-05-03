import os
import tempfile
import torch
from unittest.mock import MagicMock, patch

from src.evaluation.setup import parse_args, setup_output_directory
from src.evaluation.loading import load_model_from_checkpoint
from src.evaluation.data import get_evaluation_dataloader
from src.evaluation.core import evaluate_model
from src.utils.visualization import visualize_predictions
from src.evaluation.results import save_evaluation_results
from src.evaluation.ensemble import ensemble_evaluate


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        'sys.argv', ['evaluate.py', '--checkpoint', 'ckpt.pth.tar']
    )
    args = parse_args()
    assert args.checkpoint == 'ckpt.pth.tar'
    assert args.visualize_samples == 5


def test_setup_output_directory_creates_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = setup_output_directory(tmpdir)
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, 'metrics'))
        assert os.path.exists(os.path.join(out_dir, 'visualizations'))


@patch('src.model.factory.create_unet')
def test_load_model_from_checkpoint(mock_create_unet):
    # Crear un archivo temporal para simular un checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth') as temp_file:
        checkpoint_path = temp_file.name

        # Configurar mocks
        dummy_model = torch.nn.Linear(2, 2)
        mock_create_unet.return_value = dummy_model

        # Crear un checkpoint simulado en el archivo temporal
        dummy_state = {
            'model_state_dict': {
                'weight': torch.randn(2, 2),
                'bias': torch.randn(2)
            },
            'config': {'model': {}}
        }
        torch.save(dummy_state, checkpoint_path)

        # Patch para evitar la verificación del modelo
        with patch('torch.nn.Module.load_state_dict'):
            # Llamar a la función bajo prueba
            model, data = load_model_from_checkpoint(
                checkpoint_path, torch.device('cpu')
            )

        # Verificar los resultados
        assert isinstance(model, torch.nn.Module)
        assert model is dummy_model
        assert 'config' in data
        mock_create_unet.assert_called_once()


def test_get_evaluation_dataloader(monkeypatch):
    dummy_loader = MagicMock()
    dummy_loader.dataset = [1, 2, 3]  # Simula un dataset con 3 muestras
    monkeypatch.setattr(
        'src.evaluation.data.create_dataloaders_from_config',
        lambda **kwargs: {'test': {'dataloader': dummy_loader}}
    )
    cfg = {'data': {}}
    loader = get_evaluation_dataloader(cfg)
    assert loader is dummy_loader


def test_evaluate_model_basic():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])
    model = DummyModel()
    batch = {'image': torch.zeros(2, 3, 4, 4),
             'mask': torch.zeros(2, 1, 4, 4)}
    loader = [batch]
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}
    results, (inputs, targets, outputs) = evaluate_model(
        model, loader, metrics, torch.device('cpu')
    )
    assert 'test_dummy' in results
    assert inputs.shape[0] == 2


def test_visualize_predictions_creates_files(tmp_path):
    inputs = torch.zeros(2, 3, 4, 4)
    targets = torch.zeros(2, 1, 4, 4)
    outputs = torch.zeros(2, 1, 4, 4)
    visualize_predictions(
        inputs, targets, outputs, str(tmp_path), num_samples=1
    )
    vis_dir = os.path.join(tmp_path, 'visualizations')
    assert any(f.endswith('.png') for f in os.listdir(vis_dir))


def test_save_evaluation_results_creates_files(tmp_path):
    results = {'test_metric': 1.0}
    config = {'foo': 'bar'}
    checkpoint = 'ckpt.pth.tar'
    save_evaluation_results(
        results, config, checkpoint, str(tmp_path)
    )
    metrics_dir = os.path.join(tmp_path, 'metrics')
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.yaml'))
    assert os.path.exists(os.path.join(metrics_dir, 'evaluation_results.txt'))


def test_ensemble_evaluate_creates_results_and_files(tmp_path, monkeypatch):
    # Dummy model que siempre predice unos
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x[:, :1])

    # Mock de load_model_from_checkpoint para devolver DummyModel
    def fake_load_model_from_checkpoint(path, device):
        return DummyModel(), {'config': {'model': {}}}
    monkeypatch.setattr('src.evaluation.ensemble.load_model_from_checkpoint',
                        fake_load_model_from_checkpoint)

    # Dummy dataloader: 2 batches de 2 muestras
    batch = {'image': torch.zeros(2, 3, 4, 4), 'mask': torch.zeros(2, 1, 4, 4)}
    dataloader = [batch, batch]

    # Dummy metric
    metrics = {'dummy': lambda o, t: torch.tensor(1.0)}

    # Mock de visualize_predictions para evitar crear imágenes reales
    monkeypatch.setattr('src.evaluation.ensemble.visualize_predictions',
                        lambda *a, **kw: None)

    # Ejecutar ensemble_evaluate
    results = ensemble_evaluate(
        checkpoint_paths=['ckpt1.pth', 'ckpt2.pth'],
        config={'model': {}},
        dataloader=dataloader,
        metrics=metrics,
        device=torch.device('cpu'),
        output_dir=str(tmp_path)
    )
    # Verifica que la clave de resultado esté presente y el valor sea correcto
    assert 'ensemble_dummy' in results
    assert results['ensemble_dummy'] == 1.0

    # Verifica que se crea el archivo ensemble_results.yaml
    ensemble_dir = os.path.join(tmp_path, 'metrics', 'ensemble')
    yaml_path = os.path.join(ensemble_dir, 'ensemble_results.yaml')
    assert os.path.exists(yaml_path)
    # Verifica que el archivo contiene la métrica
    with open(yaml_path, 'r') as f:
        content = f.read()
        assert 'ensemble_dummy' in content
