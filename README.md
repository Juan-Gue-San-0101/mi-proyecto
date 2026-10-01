# Vessel v2 - Deteccion de barcos

Este proyecto usa inteligencia artificial para encontrar y marcar barcos en imagenes. 

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

pip install -r requirements.txt

## Como usarlo

### Detectar barcos en una imagen

python predict_v3.py --input ruta/a/imagen.jpg --output resultado.png

### Entrenar el modelo desde cero

python trainModelVessel_v2.py --config config.yaml

### Preparar las imagenes antes de entrenar

python build_tiles.py --input Dataset/imagenes --output Dataset/tiles
python build_masks.py --input Dataset/annotations --output Dataset/masks

### Convertir el modelo a ONNX (para que corra mas rapido)

python export_onnx.py --weights weights/model_best_ep20.pth --output model.onnx

### Ver que tan rapido funciona

python benchmark_onnx.py --model model.onnx

## Que hay en cada archivo

- Vessel_models.py: aqui estan las redes neuronales
- trainModelVessel_v2.py: entrena el modelo
- predict_v3.py: detecta barcos en imagenes nuevas
- predict_segmentation.py: otra forma de detectar
- build_tiles.py: parte imagenes grandes en pedazos
- build_masks.py: crea las mascaras para entrenar
- export_onnx.py: convierte el modelo a otro formato mas rapido
- benchmark_onnx.py: mide la velocidad del modelo
- weights/: aqui estan los modelos ya entrenados
- archivos .onnx: modelos listos para usar
- requirements.txt: lista de cosas que hay que instalar

## Modelos que vienen incluidos

- weights/model_best_ep20.pth: el mejor modelo de la vuelta 20 (13 MB)
- weights/model_final.pth: el modelo final (13 MB)
- weights/checkpoint.pth: copia de seguridad del entrenamiento (37 MB)
- model_best_ep20_fp32.onnx: modelo normal (12 MB)
- model_best_ep20_int8.onnx: modelo mas chico (12 MB)
- model_best_ep20_int8_dynamic.onnx: el mas liviano (4.2 MB)

## Sobre las imagenes de entrenamiento

Las imagenes con las que se entreno el modelo no estan aqui porque pesan mucho se necesitan descargar por separado.

## Con que esta hecho

- PyTorch
- ONNX
- OpenCV
- NumPy
- OpenVINO



