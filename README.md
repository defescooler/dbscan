# DBSCAN Classifier in C

DBSCAN clustering on pistachio images (hue/saturation features), visualized with raylib.

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
