"""Build-time support: embed the original base loader beside the vendor provider."""
import os
import shutil


def copy_loader_framework(source, build_lib):
    destination = os.path.join(build_lib, 'cann_ops_transformer')
    for root, dirs, files in os.walk(source):
        dirs[:] = [name for name in dirs if name != '__pycache__' and not name.startswith('.')]
        relative = os.path.relpath(root, source)
        first = relative.split(os.sep)[0]
        if first == 'csrc' or (first == 'ops' and relative != 'ops'):
            dirs[:] = []
            continue
        for name in files:
            if name.endswith(('.pyc','.pyo')) or name.startswith('.'):
                continue
            if first == 'ops' and name != '__init__.py':
                continue
            target = os.path.join(destination, relative, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(root, name), target)
