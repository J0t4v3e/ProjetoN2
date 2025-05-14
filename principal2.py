import dlib  # Biblioteca para reconhecimento de rostos utilizando HOG (Histograms of Oriented Gradients)
import cv2  # OpenCV, utilizado para processamento de imagens e vídeos
import matplotlib.pyplot as plt  # Usado para exibir a imagem com os resultados
import numpy as np  # Biblioteca para manipulação de arrays, não está sendo usada diretamente no código

# Carregar a imagem enviada
image_path = "lucas2.jpg"
image = cv2.imread(image_path)  # Carrega a imagem a partir do caminho especificado

# Verificar se a imagem foi carregada corretamente
if image is None:
    raise FileNotFoundError(f"Imagem não encontrada no caminho: {image_path}")  # Levanta um erro caso a imagem não seja encontrada

# Ajustar brilho e contraste automaticamente para melhorar a detecção
image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Aumenta brilho e contraste da imagem

# Aplicar suavização para reduzir ruído
image = cv2.GaussianBlur(image, (5, 5), 0)  # Aplica um filtro de suavização para reduzir ruído na imagem

# Converter a imagem para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # A detecção de rostos é geralmente feita em escala de cinza para melhorar a performance

# Inicializar o detector de rosto frontal HOG da dlib
detector_hog = dlib.get_frontal_face_detector()  # Inicializa o detector de rostos frontais usando HOG

# Inicializar o detector de rosto de perfil (Haar Cascade do OpenCV)
profile_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\data\haarcascade_profileface.xml")  # Carrega o classificador Haar Cascade para detecção de rostos de perfil

# Detectar rostos frontais (HOG) - Ajustando o nível de confiança
faces_front = detector_hog(gray_image, 1)  # Detecta rostos frontais com uma pirâmide de nível 1 para mais precisão

# Detectar rostos de perfil direito (Cascade) com parâmetros otimizados
faces_profile_right = profile_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=4,   # Reduzido para maior sensibilidade (mais deteções, mas pode aumentar falsos positivos)
    minSize=(30, 30),
    maxSize=(800, 800)  # Para não detectar falsos positivos muito grandes
)

# Detectar rostos de perfil esquerdo (espelhando a imagem)
flipped_gray = cv2.flip(gray_image, 1)  # Espelhar a imagem horizontalmente para detectar rostos de perfil esquerdo
faces_profile_left = profile_cascade.detectMultiScale(
    flipped_gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
    maxSize=(800, 800)
)

# Desenhar retângulos para os rostos frontais detectados (HOG)
for face in faces_front:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()  # Extrai as coordenadas do rosto
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha o retângulo verde para rostos frontais

# Desenhar retângulos para os rostos de perfil direito
for (x, y, w, h) in faces_profile_right:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Desenha o retângulo azul para rostos de perfil direito

# Desenhar retângulos para os rostos de perfil esquerdo (ajustando posição)
for (x, y, w, h) in faces_profile_left:
    # Corrigir as coordenadas para o espelhamento
    x = gray_image.shape[1] - x - w  # Ajusta a posição dos rostos detectados após o espelhamento da imagem
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Desenha o retângulo vermelho para rostos de perfil esquerdo

# Exibir a imagem com os rostos detectados
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Converte a imagem para RGB antes de exibir com matplotlib
plt.axis('off')  # Remove os eixos
plt.show()  # Exibe a imagem final com os rostos detectados
