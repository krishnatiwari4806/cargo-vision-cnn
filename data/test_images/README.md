# Manual Test Images Directory

This directory (`data/test_images/`) is dedicated **ONLY** for manual interactive image testing using:

```powershell
.\.venv\Scripts\python.exe run.py
```
or
```bat
test.bat
```

### Supported Image Formats:
- `.jpg` / `.jpeg`
- `.png`
- `.webp`
- `.bmp`

### Usage Notice:
- Images placed in this directory are **NOT** part of the training, validation, or test evaluation datasets.
- Files here will **NOT** be modified, moved, or deleted automatically.
- RGB photographs do not directly measure physical mass (kg). Weight from category priors is an estimated reference.
