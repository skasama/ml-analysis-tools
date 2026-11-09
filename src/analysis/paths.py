from pathlib import Path

def input_path(base, value, label):
    base, value = Path(base).expanduser().resolve(), Path(value).expanduser()
    path = value.resolve() if value.is_absolute() else (base / value).resolve()
    if not path.is_file(): raise FileNotFoundError(f"{label} was not found: {path}")
    return path

def output_path(directory, filename):
    filename = Path(filename).expanduser()
    path = filename.resolve() if filename.is_absolute() else (Path(directory) / filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
