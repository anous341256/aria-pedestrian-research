"""Isolated real-data browser test project; never edit the user's correction DB."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]/'outputs/aria_identity_editor'
dest=root/'ui_smoke'
if dest.exists():raise FileExistsError(dest)
dest.mkdir()
for name in ['input.json','features.npz','manifest.json']:
    shutil.copyfile(root/'v1'/name,dest/name)
print(dest)
