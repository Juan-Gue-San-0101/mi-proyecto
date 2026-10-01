# Vessel v2 - Deteccion de barcos

Este proyecto usa inteligencia artificial para encontrar y marcar barcos en imagenes. Esta hecho con PyTorch y se puede usar en cualquier computadora.

## Que necesitas

- Python 3.10 o mas nuevo
- pip (viene con Python)
- Si quieres entrenar tu propio modelo, una tarjeta grafica NVIDIA ayuda mucho
- Git LFS si quieres bajar los modelos ya entrenados

## Como instalarlo

### 1. Bajar el proyecto

git clone https://github.com/Juan-Gue-San-0101/mi-proyecto.git
cd mi-proyecto

Si los modelos no se bajan solos, primero instala Git LFS:

sudo apt install git-lfs
git lfs install
git lfs pull

### 2. Crear un entorno de trabajo

Esto sirve para no mezclar las cosas de este proyecto con las de tu computadora:

python3 -m venv .venv
source .venv/bin/activate

### 3. Instalar lo que hace falta

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime opencv-python numpy pillow

## Como usarlo

### Detectar barcos en una imagen

IMPORTANTE: usa siempre el modelo model_best_ep20_fp32_consolidated.onnx, los otros modelos .onnx no funcionan bien.

python3 predict_v3.py --model model_best_ep20_fp32_consolidated.onnx --input muestra_test/boat_data_ext_00061.jpg --output-dir resultado --visualization

### Detectar barcos en varias imagenes a la vez

python3 predict_v3.py --model model_best_ep20_fp32_consolidated.onnx --input muestra_test/ --output-dir resultado --visualization

### Ver los resultados

Los resultados se guardan en la carpeta que pusiste en --output-dir. Dentro encontraras:

- Un archivo _overlay.jpg con los barcos marcados encima de la imagen
- Un archivo _boxes.csv con las coordenadas de cada barco
- Un archivo detections.json con toda la informacion

Para abrir la imagen con los barcos marcados:

xdg-open resultado/boat_data_ext_00061_overlay.jpg

### Entrenar el modelo desde cero

python trainModelVessel_v2.py --config config.yaml

### Preparar las imagenes antes de entrenar

python build_tiles.py --input Dataset/imagenes --output Dataset/tiles
python build_masks.py --input Dataset/annotations --output Dataset/masks

### Convertir el modelo a ONNX

python export_onnx.py --weights weights/model_best_ep20.pth --output model.onnx

## Modelos que vienen incluidos

- model_best_ep20_fp32_consolidated.onnx: modelo completo. FUNCIONA BIEN Y ES RAPIDO. Usa este.
- model_best_ep20_fp32.onnx: modelo completo en partes (necesita el archivo .data)
- model_best_ep20_int8_consolidated.onnx: modelo cuantizado, sin verificar
- model_best_ep20_int8.onnx: modelo cuantizado en partes
- model_best_ep20_int8_dynamic.onnx: NO USAR. No detecta nada, esta roto.
- weights/model_best_ep20.pth: modelo original de PyTorch (13 MB)
- weights/model_final.pth: modelo final de PyTorch (13 MB)
- weights/checkpoint.pth: copia de seguridad del entrenamiento (37 MB)

## Que hay en cada archivo

- Vessel_models.py: aqui estan las redes neuronales
- trainModelVessel_v2.py: entrena el modelo
- predict_v3.py: detecta barcos en imagenes nuevas
- predict_segmentation.py: otra forma de detectar
- build_tiles.py: parte imagenes grandes en pedazos
- build_masks.py: crea las mascaras para entrenar
- export_onnx.py: convierte el modelo a otro formato
- benchmark_onnx.py: mide la velocidad del modelo
- weights/: aqui estan los modelos ya entrenados
- requirements.txt: lista de cosas que hay que instalar

## Sobre las imagenes de entrenamiento

Las imagenes con las que se entreno el modelo no estan aqui porque pesan mucho. Si las necesitas, escribeme.

## Con que esta hecho

- PyTorch
- ONNX
- OpenCV
- NumPy
- OpenVINO




