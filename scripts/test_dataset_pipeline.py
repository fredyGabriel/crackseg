import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(" PYTHONPATH ajustado.")

try:
    from omegaconf import OmegaConf
    print(" OmegaConf importado.")
    from src.data import create_crackseg_dataset
    print(" create_crackseg_dataset importado.")
except ImportError as e:
    print(f"Error importando: {e}", file=sys.stderr)
    sys.exit(1)

# Simulación de lista de muestras (debes reemplazar con rutas reales para un
# test real)
samples_list = [
    ("data/dummy_img.png", "data/dummy_mask.png"),
    # Puedes añadir más si quieres probar la longitud
    # ("data/dummy_img.png", "data/dummy_mask.png"),
]
print(" Lista de muestras definida.")


def main():
    print(" Entrando a main().")
    # Cargar configuraciones
    try:
        print(" Cargando data_cfg...")
        data_cfg = OmegaConf.load("configs/data/default.yaml")
        print(" data_cfg cargado.")
        print(" Cargando transform_cfg...")
        transform_cfg = OmegaConf.load("configs/data/transform.yaml")
        print(" transform_cfg cargado.")
    except Exception as e:
        print(f"Error cargando configs: {e}", file=sys.stderr)
        sys.exit(1)

    # Crear dataset para modo 'train'
    try:
        print(" Llamando a create_crackseg_dataset...")
        dataset = create_crackseg_dataset(
            data_cfg=data_cfg,
            transform_cfg=transform_cfg,
            mode="train",
            samples_list=samples_list
        )
        print(" create_crackseg_dataset ejecutado.")
        print(
            f"Dataset creado correctamente. Número de muestras: {len(dataset)}"
        )
        print(" Accediendo a la muestra 0...")
        sample = dataset[0]
        print(" Muestra 0 accedida.")
        print(
            f"Shape imagen: {sample['image'].shape}, "
            f"Shape máscara: {sample['mask'].shape}"
        )
    except Exception as e:
        print(f"Error al crear o usar el dataset: {e}", file=sys.stderr)
        sys.exit(1)
    print(" main() finalizado.")


if __name__ == "__main__":
    print(" Ejecutando script...")
    main()
    print(" Script finalizado.")
