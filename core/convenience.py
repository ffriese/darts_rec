import cv2 as cv

from core.datatypes import CVImage


def resize(image, factor=1, shape=None):
    if shape is None:
        shape = image.shape
    return CVImage(cv.resize(image, (int(shape[1]*factor), int(shape[0]*factor))),
                   image.id, image.camera_info)