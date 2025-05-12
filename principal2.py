import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregar a imagem enviada
image_path = "lucas2.jpg"
image = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")

# Ajustar brilho e contraste automaticamente para melhorar a detecção
image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Aumenta brilho e contraste

# Aplicar suavização para reduzir ruído
image = cv2.GaussianBlur(image, (5, 5), 0)

# Converter a imagem para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Inicializar o detector de rosto frontal HOG da dlib
detector_hog = dlib.get_frontal_face_detector()

# Inicializar o detector de rosto de perfil (Haar Cascade do OpenCV)
profile_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\data\haarcascade_profileface.xml")

# Detectar rostos frontais (HOG) - Ajustando o nível de confiança
faces_front = detector_hog(gray_image, 1)  # Nível de pirâmide 1 para mais precisão

# Detectar rostos de perfil direito (Cascade) com parâmetros otimizados
faces_profile_right = profile_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,   # Reduzido para maior sensibilidade
    minSize=(30, 30),
    maxSize=(800, 800)  # Para não detectar falsos positivos muito grandes
)

# Detectar rostos de perfil esquerdo (espelhando a imagem)
flipped_gray = cv2.flip(gray_image, 1)  # Espelhar horizontalmente
faces_profile_left = profile_cascade.detectMultiScale(
    flipped_gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# Desenhar retângulos para os rostos frontais detectados (HOG)
for face in faces_front:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde para rostos frontais

# Desenhar retângulos para os rostos de perfil direito
for (x, y, w, h) in faces_profile_right:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Azul para perfil direito

# Desenhar retângulos para os rostos de perfil esquerdo (ajustando posição)
for (x, y, w, h) in faces_profile_left:
    # Corrigir as coordenadas para o espelhamento
    x = gray_image.shape[1] - x - w
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vermelho para perfil esquerdo

# Exibir a imagem com os rostos detectados
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
