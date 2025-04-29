import os
import tempfile
from src.utils import env


def test_load_env_and_get_env_var():
    # Crear un archivo .env temporal
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write('TEST_KEY=hello_world\n')
        tmp_path = tmp.name
    try:
        env.load_env(tmp_path)
        assert env.get_env_var('TEST_KEY') == 'hello_world'
        assert env.get_env_var('NON_EXISTENT', 'default') == 'default'
    finally:
        os.remove(tmp_path)
