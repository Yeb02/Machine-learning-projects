import face_recognition as fr
import os
import cv2
import numpy as np
from time import sleep


def get_encoded_faces(path):
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}
    print(os.walk(path))
    for dirpath, dnames, fnames in os.walk(path):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(path + f)
                print(face.shape)

                en = fr.face_encodings(face)
                print(len(en))

                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

u = get_encoded_faces('./faces/')
