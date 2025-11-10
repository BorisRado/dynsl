from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ControlCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OK: _ClassVar[ControlCode]
    DO_CLOSE_STREAM: _ClassVar[ControlCode]
    STREAM_CLOSED_OK: _ClassVar[ControlCode]
    ERROR_PROCESSING_STREAM: _ClassVar[ControlCode]
    INIT_STREAM: _ClassVar[ControlCode]
OK: ControlCode
DO_CLOSE_STREAM: ControlCode
STREAM_CLOSED_OK: ControlCode
ERROR_PROCESSING_STREAM: ControlCode
INIT_STREAM: ControlCode

class BatchDataForward(_message.Message):
    __slots__ = ("training", "data", "control_code", "cid", "local_computation_time", "tot_local_time")
    class DataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ByteTensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ByteTensor, _Mapping]] = ...) -> None: ...
    TRAINING_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_COMPUTATION_TIME_FIELD_NUMBER: _ClassVar[int]
    TOT_LOCAL_TIME_FIELD_NUMBER: _ClassVar[int]
    training: bool
    data: _containers.MessageMap[str, ByteTensor]
    control_code: ControlCode
    cid: str
    local_computation_time: float
    tot_local_time: float
    def __init__(self, training: bool = ..., data: _Optional[_Mapping[str, ByteTensor]] = ..., control_code: _Optional[_Union[ControlCode, str]] = ..., cid: _Optional[str] = ..., local_computation_time: _Optional[float] = ..., tot_local_time: _Optional[float] = ...) -> None: ...

class BatchDataBackward(_message.Message):
    __slots__ = ("data", "control_code", "layers_diff", "additional_layers_parameters", "training_loss")
    class DataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ByteTensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ByteTensor, _Mapping]] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    CONTROL_CODE_FIELD_NUMBER: _ClassVar[int]
    LAYERS_DIFF_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_LAYERS_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_LOSS_FIELD_NUMBER: _ClassVar[int]
    data: _containers.MessageMap[str, ByteTensor]
    control_code: ControlCode
    layers_diff: int
    additional_layers_parameters: TensorList
    training_loss: float
    def __init__(self, data: _Optional[_Mapping[str, ByteTensor]] = ..., control_code: _Optional[_Union[ControlCode, str]] = ..., layers_diff: _Optional[int] = ..., additional_layers_parameters: _Optional[_Union[TensorList, _Mapping]] = ..., training_loss: _Optional[float] = ...) -> None: ...

class ByteTensor(_message.Message):
    __slots__ = ("single_tensor", "tensors")
    SINGLE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    single_tensor: bytes
    tensors: TensorList
    def __init__(self, single_tensor: _Optional[bytes] = ..., tensors: _Optional[_Union[TensorList, _Mapping]] = ...) -> None: ...

class TensorList(_message.Message):
    __slots__ = ("tensors",)
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, tensors: _Optional[_Iterable[bytes]] = ...) -> None: ...
