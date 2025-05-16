import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if (len(rostos) > 0):
        return True, rostos
    
    return False, []

def get_rostos():

    rostos_conhecidos = []
    nomes_dos_rostos = []

    imagem1 = reconhece_face("./img/luiz1.jpg")
    if (imagem1[0]):
        rostos_conhecidos.append(imagem1[1][0])
        nomes_dos_rostos.append("Luiz")

    imagem3 = reconhece_face("./img/du.jpg")
    if (imagem3[0]):
        rostos_conhecidos.append(imagem3[1][0])
        nomes_dos_rostos.append("Du")
    return rostos_conhecidos, nomes_dos_rostos
