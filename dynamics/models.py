from dynamics.cuda_dynamics import *
from dynamics.native_dynamics import *
from dataclasses import dataclass

@dataclass
class DynamicsModel:
    gpu: callable
    cpu: callable
    metadata: dict

bicycle_model = DynamicsModel(
    gpu=bicycle_dynamics,
    cpu=bicycle_dynamics_host,
    metadata=bicycle_dynamics.metadata 
)

diffdrive_model = DynamicsModel(
    gpu=differential_drive,
    cpu=differential_drive_host,
    metadata=differential_drive.metadata
)

ackermann_model = DynamicsModel(
    gpu=ackermann_dynamics,
    cpu=ackermann_dynamics_host,
    metadata=ackermann_dynamics.metadata
)

DYNAMICS_REGISTRY = {
    "bicycle": bicycle_model,
    "differential_drive": diffdrive_model,
    "ackermann": ackermann_model
}