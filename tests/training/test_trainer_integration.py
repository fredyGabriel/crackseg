import torch
import shutil
from omegaconf import OmegaConf
from src.utils.logging import NoOpLogger
from src.training.trainer import Trainer


def get_dummy_data_loader(num_batches=4, batch_size=2, shape=(3, 4, 4)):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return num_batches * batch_size

        def __getitem__(self, idx):
            return {'image': torch.randn(*shape), 'mask': torch.randn(1, 4, 4)}
    return torch.utils.data.DataLoader(DummyDataset(), batch_size=batch_size)


def get_dummy_model():
    return torch.nn.Conv2d(3, 1, 1)


def get_dummy_loss():
    return torch.nn.MSELoss()


def get_dummy_metrics():
    return {}


def integration_test_trainer_checkpoint_resume(tmp_path, use_amp=False,
                                               grad_accum_steps=1):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({
        'training': {
            'epochs': 2,
            'device': 'cpu',
            'use_amp': use_amp,
            'gradient_accumulation_steps': grad_accum_steps,
            'checkpoint_dir': str(checkpoint_dir),
            'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.01},
            'scheduler': {
                '_target_': 'torch.optim.lr_scheduler.StepLR',
                'step_size': 1,
                'gamma': 0.5
            },
            'verbose': False,
            'log_interval_batches': 0
        }
    })

    model = get_dummy_model()
    train_loader = get_dummy_data_loader()
    val_loader = get_dummy_data_loader()
    loss_fn = get_dummy_loss()
    metrics = get_dummy_metrics()
    logger = NoOpLogger()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics,
        cfg=cfg,
        logger_instance=logger
    )
    trainer.train()

    last_ckpt = checkpoint_dir / "checkpoint_last.pth"
    assert last_ckpt.exists(), "No se guardó el checkpoint"

    # Carga desde el checkpoint y continúa entrenamiento
    cfg.training['checkpoint_load_path'] = str(last_ckpt)
    cfg.training['epochs'] = 3  # Entrenar una época más

    model2 = get_dummy_model()
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics,
        cfg=cfg,
        logger_instance=logger
    )
    trainer2.train()

    assert trainer2.start_epoch == 3, f"El entrenamiento no continuó desde la \
época correcta: {trainer2.start_epoch}"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert not torch.equal(p1, p2), "Los pesos del modelo no cambiaron \
tras reanudar y entrenar"

    shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_trainer_integration_checkpoint_resume_cpu(tmp_path):
    integration_test_trainer_checkpoint_resume(tmp_path, use_amp=False,
                                               grad_accum_steps=1)


def test_trainer_integration_checkpoint_resume_amp(tmp_path):
    integration_test_trainer_checkpoint_resume(tmp_path, use_amp=True,
                                               grad_accum_steps=1)


def test_trainer_integration_checkpoint_resume_accum(tmp_path):
    integration_test_trainer_checkpoint_resume(tmp_path, use_amp=False,
                                               grad_accum_steps=2)
