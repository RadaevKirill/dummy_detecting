# https://docs.openvino.ai/nightly/openvino_2_0_inference_pipeline.html
# https://docs.openvino.ai/latest/omz_demos_object_detection_demo_python.html
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Optional

import cv2  # type: ignore
import numpy as np
from nptyping import NDArray, Shape
from openvino.runtime import Core  # type: ignore

from example.models import Image

input_type_mapper = {'uint8': np.uint8, 'float32': np.float32}
input_dims_mapper = {ord('b'): '0', ord('c'): '1', ord('h'): '2', ord('w'): '3'}
Blob = NDArray[Shape['*, *, *, *'], Any]


class OpenvinoAdapterMixin:
    def __init__(
            self,
            device: str,
            performance_hint: str,
            execution_mode_hint: str,
            inference_precision_hint: str,
            model: str,
            weights: str,
            input_type: str,
            input_dims_order: str,
            swap_rb: bool,
            height: int,
            width: int,
            score_threshold: float,
            nms_threshold: float,
    ):
        if performance_hint not in ['THROUGHPUT', 'LATENCY']:
            raise ValueError('Performance hint should be one of ["LATENCY", "THROUGHPUT"]')

        if input_type.lower() not in input_type_mapper.keys():
            raise ValueError(f'Input type should be one of {list(input_type_mapper.keys())}')

        if (len(set(input_dims_order.lower())) != 4) or (set('bchw') - set(input_dims_order.lower())):
            raise ValueError('Invalid order of input dimensions')

        if not Path(model).is_file():
            raise FileNotFoundError(f'Model file "{model}" not found')

        if not Path(weights).is_file():
            raise FileNotFoundError(f'Weights file "{weights}" not found')

        logging.info('Creating inference engine')
        self.__core = Core()
        device = device.upper() if device.upper() in [x.upper() for x in self.__core.available_devices] else 'CPU'

        logging.info(f'Use model: {model}')
        logging.info(f'Loading network to {device}')
        logging.info(f'PERFORMANCE_HINT: {performance_hint}')
        logging.info(f'EXECUTION_MODE_HINT: {execution_mode_hint}')
        logging.info(f'INFERENCE_PRECISION_HINT: {inference_precision_hint}')

        self.__compiled_model = self.__core.compile_model(
            model=str(model),
            device_name=device,
            config={
                'PERFORMANCE_HINT': performance_hint,
                'EXECUTION_MODE_HINT': execution_mode_hint,
                'INFERENCE_PRECISION_HINT': inference_precision_hint,
            },
        )
        self.__infer_request = self.__compiled_model.create_infer_request()

        self.__input_type: type = input_type_mapper[input_type.lower()]
        self.__input_dims_transpose_list: List[int] = \
            list(map(int, input_dims_order.lower().translate(input_dims_mapper)))
        self.__swap_rb: bool = swap_rb
        self._height: int = height
        self._width: int = width

        self._score_threshold: float = score_threshold
        self._nms_threshold: float = nms_threshold

        logging.info(f'Image input height: {self._height}')
        logging.info(f'Image input width: {self._width}')

    def __pre_processing(self, image: Any) -> Blob:
        return np.transpose(
            a=cv2.dnn.blobFromImage(
                image=image,
                size=(self._width, self._height),
                swapRB=self.__swap_rb,
            ),
            axes=self.__input_dims_transpose_list,
        ).astype(self.__input_type)

    def __infer(self, data: Blob) -> Any:
        self.__infer_request.infer([data])
        return np.array(list(self.__infer_request.results.values())[0])

    @abstractmethod
    def _post_processing(self, output: Any, image_height: int, image_width: int) -> Any:
        raise NotImplementedError

    def _predict(self, image: Image) -> Any:
        input_ = self.__pre_processing(image=image)
        output = self.__infer(data=input_)
        result = self._post_processing(output=output, image_height=image.shape[0], image_width=image.shape[1])
        return result
