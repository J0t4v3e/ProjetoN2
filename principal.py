import cv2
import sys

# Inicializa o descritor HOG e define o detector de pessoas padrão
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Carrega a imagem com caminho absoluto
image = cv2.imread('C:/Users/natan/.spyder-py3/casal.jpeg')
if image is None:
    print("Erro: A imagem não foi carregada corretamente.")
    sys.exit()

# Detecta pessoas na imagem
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)

# Desenha retângulos ao redor das pessoas detectadas
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Exibe a imagem resultante
cv2.imshow("Detecções", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
