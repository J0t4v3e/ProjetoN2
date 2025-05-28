#LINK DE ESTUDO PARA IDENTIFICAR IMAGENS ESCURAS https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time  # biblioteca para cálculo de tempo

# Caminho da imagem
image_path = "front.jpg"
image = cv2.imread(image_path)# lendo a imagem

if image is None:
    raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")#confirmando se a imagem esta carregada

# processando a imagem
image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)#ajustando o brilho
image = cv2.GaussianBlur(image, (5, 5), 0)#aplicando efeito gaussiano (AUMENTAR O CONTRASTE)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#ajustando a escala de cinza

#detector de rostos frontais com HOG da bibliteca dlib
detector_hog = dlib.get_frontal_face_detector()

# classifica Haarcascade com um xml ja treinado(perfis)
profile_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\data\haarcascade_profileface.xml")

if profile_cascade.empty():
    raise IOError("Erro ao carregar o haarcascade_profileface.xml")#verificando se carregou com sucesso 

# star de time na execução
start_time = time.time()

# detectar os rostos
faces_front = detector_hog(gray_image, 1)

#detectando rostos de lado direito
faces_profile_right = profile_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# espelhando para detectar perfil esquerdo
flipped_gray = cv2.flip(gray_image, 1)
faces_profile_left = profile_cascade.detectMultiScale(
    flipped_gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# para o tempo
end_time = time.time()

#calcula a diferença de tempo
detection_time = end_time - start_time

# printa o tempo, quantidade de rostos frontais, direitos e esquerdos
print(f"Tempo de detecção: {detection_time:.2f} segundos")
print(f"Rostos frontais detectados: {len(faces_front)}")
print(f"Perfis direitos detectados: {len(faces_profile_right)}")
print(f"Perfis esquerdos detectados: {len(faces_profile_left)}")

# colocando os retangulos em volta do rosto
for face in faces_front:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Frontal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
# colocando os retangulos em volta do rosto direito
for (x, y, w, h) in faces_profile_right:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, "Perfil Dir", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)
# colocando os retangulos em volta do esquerdo
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
