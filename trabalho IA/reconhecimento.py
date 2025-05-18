import cv2
import time
import face_recognition as fr
from mtcnn import MTCNN
from engine import reconhece_face, get_rostos

# Início do processo
inicio = time.time()

# Carrega imagem
imagem_rgb = fr.load_image_file("./img/oculos3.jpg")
imagem_bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)

# Detecta rostos com MTCNN (visualização lateral)
detector = MTCNN()
detecoes = detector.detect_faces(imagem_rgb)

if not detecoes:
    print("Nenhum rosto detectado.")
else:
    rostos_conhecidos, nomes_dos_rostos = get_rostos()

    for det in detecoes:
        x, y, largura, altura = det['box']
        x, y = max(0, x), max(0, y)
        top, right, bottom, left = y, x + largura, y + altura, x

        # Gera a codificação para esse rosto
        face_encoding = fr.face_encodings(imagem_rgb, [(top, right, bottom, left)])
        if not face_encoding:
            continue

        encoding_desconhecido = face_encoding[0]
        resultados = fr.compare_faces(rostos_conhecidos, encoding_desconhecido)
        nome = "Desconhecido"

        for i, resultado in enumerate(resultados):
            if resultado:
                nome = nomes_dos_rostos[i]
                print("Rosto do", nome, "foi reconhecido")
                break

        # Desenha caixa e nomes
        cv2.rectangle(imagem_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imagem_bgr, nome, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra o tempo de execução do programa
    fim = time.time()
    print(f"\nTempo total de reconhecimento: {fim - inicio:.2f} segundos")

    # Redimensiona imagem
    largura_max, altura_max = 1280, 720
    altura, largura = imagem_bgr.shape[:2]
    escala = min(largura_max / largura, altura_max / altura, 1.0)
    nova_largura, nova_altura = int(largura * escala), int(altura * escala)
    imagem_redimensionada = cv2.resize(imagem_bgr, (nova_largura, nova_altura))

    # Exibe imagem
    cv2.imshow("Reconhecimento Facial", imagem_redimensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
