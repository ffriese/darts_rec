import uuid
from typing import List, Tuple

import numpy as np


class AbstractRecognitionDataType(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.source = None
        return obj


class RecognitionDataType(object):
    __metaclass__ = AbstractRecognitionDataType
    source = None


class CVImage(np.ndarray, RecognitionDataType):

    def __new__(cls, input_array, _id, camera_info):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if _id is None:
            _id = uuid.uuid4()
        obj.id = _id
        obj.camera_info = camera_info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.id = getattr(obj, 'id', None)
        self.camera_info = getattr(obj, 'camera_info', None)

    def cam_id(self):
        return self.camera_info['name']

    def __str__(self):
        return 'CVImage[%s, %r, %r]' % (getattr(self, 'id', None), getattr(self, 'camera_info', None), self.shape)


class MultiImage(RecognitionDataType):
    def __init__(self, images: List[CVImage], processing_trigger=False):
        self.images = images
        self.has_processing_trigger = processing_trigger

    def __len__(self):
        return len(self.images)


class Contours(RecognitionDataType):
    def __init__(self, contours, image_id, camera_info):
        self.image_id = image_id
        self.contours = contours
        self.camera_info = camera_info


class ContourCollection(RecognitionDataType):
    def __init__(self, contour_collection: List[Contours]):
        self.collection = contour_collection


class ImpactPoint(RecognitionDataType):
    def __init__(self, point: Tuple[int, int], image_id: str, camera_info):
        self.point = point
        self.image_id = image_id
        self.camera_info = camera_info


class ImpactPoints(RecognitionDataType):
    def __init__(self, points: List[ImpactPoint]):
        self.points = points


class JsonObject(RecognitionDataType):
    def __init__(self, json, topic):
        self.json = json
        self.topic = topic


class SetBackgroundTrigger(RecognitionDataType):
    def __init__(self, dart_number: int):
        self.dart_number = dart_number


class BoardCoordinate(RecognitionDataType):
    def __init__(self, point):
        self.point = point


class CollectionTrigger(RecognitionDataType):
    def __init__(self, processing_trigger=False):
        self.is_processing_trigger = processing_trigger
