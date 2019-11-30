from collections import deque
from enum import Enum
from threading import Lock
from typing import List

import numpy as np

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, ImpactPoint, SetBackgroundTrigger, ContourCollection
from core.convenience import resize
import cv2 as cv


class DARTS_STATES(Enum):
    IDLE = 0
    DART_1 = 1
    DART_2 = 2
    DART_3 = 3
    TAKE_OUT = 4


class StateMachine(Module):
    def __init__(self):
        super().__init__()
        self.contours_in = Input(data_type=Contours, config_keys=['cam_ids'])
        self.contour_collection_out = Output(data_type=ContourCollection, config_keys=['cam_ids'])
        self.set_background_trigger_out = Output(data_type=SetBackgroundTrigger)
        self.cam_ids = ModuleParameter(None, data_type=list)

        self.internal_state = DARTS_STATES.IDLE

        self.wait_for_matching_images = False
        self.background_reset = False
        self.match_lock = Lock()
        self.matches = {}

    def start_waiting_for_matches(self):
        self.wait_for_matching_images = True
        self.matches = {}

    def process_contours_in(self, contours):
        largest = sorted(contours.contours, key=self.a_len, reverse=True)[:10]

        if largest:
            ll = self.a_len(largest[0])
            # STATE_PROGRESSION
            if self.internal_state == DARTS_STATES.IDLE:
                self.internal_state = DARTS_STATES.DART_1
                self.start_waiting_for_matches()
                self.log_debug(ll, '->', self.internal_state)
            elif self.internal_state == DARTS_STATES.DART_1:
                if not self.wait_for_matching_images:
                    if self.background_reset:
                        self.set_background_trigger_out.data_ready(SetBackgroundTrigger(1))
                        self.background_reset = False
                    else:
                        self.internal_state = DARTS_STATES.DART_2
                        self.start_waiting_for_matches()
                        self.log_debug(ll, '->', self.internal_state)
            elif self.internal_state == DARTS_STATES.DART_2:
                if not self.wait_for_matching_images:
                    if self.background_reset:
                        self.set_background_trigger_out.data_ready(SetBackgroundTrigger(2))
                        self.background_reset = False
                    else:
                        self.internal_state = DARTS_STATES.DART_3
                        self.start_waiting_for_matches()
                        self.log_debug(ll, '->', self.internal_state)
            elif self.internal_state == DARTS_STATES.DART_3:
                if not self.wait_for_matching_images:
                    if self.background_reset:
                        self.set_background_trigger_out.data_ready(SetBackgroundTrigger(0))
                        self.background_reset = False
                    else:
                        self.internal_state = DARTS_STATES.TAKE_OUT
                        self.log_debug(ll, '->', self.internal_state)

            # COLLECT MATCHES
            with self.match_lock:
                if self.wait_for_matching_images:
                    if contours.image_id not in self.matches:
                        self.matches[contours.image_id] = {}
                    self.matches[contours.image_id][contours.camera_info['name']] = contours

                    # CHECK FOR MATCH
                    for match in self.matches:
                        # self.log_debug(len(self.matches[match].keys()), 'matches already')
                        if len(self.matches[match].keys()) == len(self.cam_ids):
                            contour_collection = []
                            for cam in self.matches[match]:
                                contour_collection.append(self.matches[match][cam])
                            self.contour_collection_out.data_ready(ContourCollection(contour_collection))
                            self.wait_for_matching_images = False
                            self.background_reset = True
        else:
            if self.internal_state == DARTS_STATES.TAKE_OUT:
                self.internal_state = DARTS_STATES.IDLE
                self.log_debug('NO CONTOURS', '->', self.internal_state)


    @staticmethod
    def a_len(c):
        return cv.arcLength(c, True)



