<!-- markdownlint-disable-file -->
# Hydra Smoke Test

- Command: `['C:\\Users\\fgrv\\miniconda3\\envs\\crackseg\\python.exe', 'C:\\Users\\fgrv\\Dev\\CursorProjects\\crackseg\\run.py', '--config-name=basic_verification']`
- CWD: `C:\Users\fgrv\Dev\CursorProjects\crackseg`
- Exit code: 1
- Elapsed: 3.83s

## Stdout

```text
[2025-08-10 01:54:13,282][crackseg.runner][INFO] - Importing main function from src.main module...
[2025-08-10 01:54:15,974][crackseg.runner][ERROR] - Import error: cannot import name 'save_checkpoint' from partially initialized module 'crackseg.utils.storage' (most likely due to a circular import) (C:\Users\fgrv\Dev\CursorProjects\crackseg\src\crackseg\utils\storage\__init__.py)
[2025-08-10 01:54:15,974][crackseg.runner][ERROR] - This typically indicates missing dependencies or incorrect installation.
[2025-08-10 01:54:15,974][crackseg.runner][ERROR] - Please verify that all dependencies are installed correctly:
[2025-08-10 01:54:15,975][crackseg.runner][ERROR] -   1. Check environment.yml or requirements.txt
[2025-08-10 01:54:15,975][crackseg.runner][ERROR] -   2. Ensure PyTorch is installed with CUDA support if needed
[2025-08-10 01:54:15,977][crackseg.runner][ERROR] -   3. Verify all src/ modules are available
```

## Stderr

```text
2025-08-10 01:54:12,966 - crackseg.runner - INFO - Starting crack segmentation training runner...
2025-08-10 01:54:12,966 - crackseg.runner - INFO - Python version: 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:29:09) [MSC v.1943 64 bit (AMD64)]
2025-08-10 01:54:12,966 - crackseg.runner - INFO - Working directory: C:\Users\fgrv\Dev\CursorProjects\crackseg
```