import face_recognition as fr
import cv2
import time
from engine import reconhece_face, get_rostos


# Marca o início do processo
inicio = time.time()

# Carrega a imagem
imagem_desconhecida = fr.load_image_file("./img/desconhecido.png")
imagem_desconhecida_bgr = cv2.cvtColor(imagem_desconhecida, cv2.COLOR_RGB2BGR)

# Detecta rostos na imagem
localizacoes = fr.face_locations(imagem_desconhecida)
encodings = fr.face_encodings(imagem_desconhecida, localizacoes)

if len(encodings) == 0:
    print("Não foi encontrado nenhum rosto")
else:
    rostos_conhecidos, nomes_dos_rostos = get_rostos()

    for (top, right, bottom, left), encoding_desconhecido in zip(localizacoes, encodings):
        resultados = fr.compare_faces(rostos_conhecidos, encoding_desconhecido)
        nome = "Desconhecido"

        for i, resultado in enumerate(resultados):
            if resultado:
                nome = nomes_dos_rostos[i]
                print("Rosto do", nome, "foi reconhecido")
                break  # para no primeiro reconhecido

        # Desenha retângulo e nome (mesmo se for "Desconhecido")
        cv2.rectangle(imagem_desconhecida_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imagem_desconhecida_bgr, nome, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Marca o fim do processo
    fim = time.time()
    duracao = fim - inicio
    print(f"\nTempo total de reconhecimento: {duracao:.2f} segundos")
    
    # Exibe pop-up com os rostos
    cv2.imshow("Reconhecimento Facial", imagem_desconhecida_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()