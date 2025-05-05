import sys
import os


# Añadir el directorio raíz al path para importar desde src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.encoder.swin_transformer_encoder import SwinTransformerEncoder


def print_model_structure():
    """Imprime la estructura completa del modelo."""
    print("Initializing SwinTransformerEncoder...")
    try:
        encoder = SwinTransformerEncoder(
            in_channels=3,
            model_name="swinv2_tiny_window16_256",
            pretrained=False,
            freeze_layers="all"  # Congelar todo inicialmente
        )
        print("Model initialized successfully!")

        # Imprimir estructura básica
        print("\nModel basic properties:")
        print(f"- Has 'stages' attribute: {hasattr(encoder.swin, 'stages')}")
        if hasattr(encoder.swin, 'stages'):
            print(f"- Number of stages: {len(encoder.swin.stages)}")

        # Analizar todos los parámetros
        print("\nCounting parameters:")
        all_params = list(encoder.swin.named_parameters())
        print(f"- Total parameter count: {len(all_params)}")

        # Encontrar prefijos únicos
        prefixes = set()
        for name, _ in all_params:
            main_part = name.split('.')[0]
            prefixes.add(main_part)

        print("\nUnique top-level parameter groups:")
        for prefix in sorted(prefixes):
            # Contar parámetros en este grupo
            count = sum(1 for name, _ in all_params if name.split('.')[0] ==
                        prefix)
            print(f"- {prefix}: {count} parameters")

            # Para el primer grupo, mostrar ejemplos
            if count > 0:
                examples = [name for name, _ in all_params if name.split('.')[0] == prefix][:3]
                for ex in examples:
                    print(f"  - Example: {ex}")

        # Buscar específicamente cualquier parámetro que contenga "stages"
        print("\nSearching for parameters containing 'stages':")
        stages_params = [name for name, _ in all_params if "stages" in name]
        if stages_params:
            print(f"- Found {len(stages_params)} parameters with 'stages'")
            for ex in stages_params[:5]:  # Mostrar los primeros 5 ejemplos
                print(f"  - {ex}")
        else:
            print("- No parameters containing 'stages' found")

        # Buscar patrones alternativos que podrían representar etapas/bloques
        print("\nSearching for alternative stage/block patterns:")
        patterns = ["stage", "block", "layer"]
        for pattern in patterns:
            matching = [name for name, _ in all_params if pattern in name.lower()]
            if matching:
                print(f"- '{pattern}': {len(matching)} parameters")
                for ex in matching[:3]:  # Mostrar algunos ejemplos
                    print(f"  - {ex}")

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nSwinTransformerEncoder Structure Analysis")
    print("=" * 50)
    print_model_structure()
    print("\nAnalysis complete")
