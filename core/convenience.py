import cv2 as cv

from core.datatypes import CVImage


def resize(image, factor):
    return CVImage(cv.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor))),
                   image.id, image.camera_info)