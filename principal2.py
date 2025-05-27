import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time  # biblioteca para cálculo de tempo

# Caminho da imagem
image_path = "front.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")

# Pré-processamento da imagem
image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
image = cv2.GaussianBlur(image, (5, 5), 0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detector HOG da dlib (rostos frontais)
detector_hog = dlib.get_frontal_face_detector()

# Classificador Haarcascade (perfis)
profile_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\data\haarcascade_profileface.xml")

if profile_cascade.empty():
    raise IOError("Erro ao carregar o haarcascade_profileface.xml")

# Iniciando o tempo de execução
start_time = time.time()

# Detecção de rostos
faces_front = detector_hog(gray_image, 1)

faces_profile_right = profile_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# Espelhando para detectar perfil esquerdo
flipped_gray = cv2.flip(gray_image, 1)
faces_profile_left = profile_cascade.detectMultiScale(
    flipped_gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# Finalizando o tempo
end_time = time.time()
detection_time = end_time - start_time

# Feedback no terminal
print(f"Tempo de detecção: {detection_time:.2f} segundos")
print(f"Rostos frontais detectados: {len(faces_front)}")
print(f"Perfis direitos detectados: {len(faces_profile_right)}")
print(f"Perfis esquerdos detectados: {len(faces_profile_left)}")

# Desenhando retângulos e labels
for face in faces_front:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Frontal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

for (x, y, w, h) in faces_profile_right:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, "Perfil Dir", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)

for (x, y, w, h) in faces_profile_left:
    x_flipped = gray_image.shape[1] - x - w  # Corrige a posição no flip
    cv2.rectangle(image, (x_flipped, y), (x_flipped + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, "Perfil Esq", (x_flipped, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

# Exibir imagem
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Salvar imagem resultado (opcional)
cv2.imwrite("resultado_detectado.jpg", image)
