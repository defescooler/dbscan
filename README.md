# DBSCAN Classifier in C

DBSCAN clustering on pistachio images (hue/saturation features), visualized with raylib.

<img width="1012" height="840" alt="image" src="https://github.com/user-attachments/assets/f772eb8a-0dff-464d-888e-5c90bed5d9c1" />


## Build

```
cc -o nob nob.c && ./nob
```

On macOS install raylib via `brew install raylib`. On Linux the bundled `raylib-5.5_linux_amd64/` is used automatically.

## Run

```
./main <kirmizi_path> <siirt_path>
```

Dataset: https://www.muratkoklu.com/datasets/

Controls: `space` run · `+`/`-` eps · `T` toggle truth · `V` toggle test points
