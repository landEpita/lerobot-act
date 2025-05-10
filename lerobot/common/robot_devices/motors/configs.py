# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False

@MotorsBusConfig.register_subclass("modbus_rtu")
@dataclass
class ModbusRTUMotorsBusConfig(MotorsBusConfig):
    port: str  # Ex: /dev/ttyUSB1 pour votre adaptateur USB-RS485
    motors: dict[str, tuple[int, str]] # Ex: {"rail_joint": (201, "NEMA17_MKS42D")} (ID esclave Modbus, nom du modèle)
    baudrate: int = 115200
    stopbits: int = 1
    parity: str = 'N' # 'N' for None, 'E' for Even, 'O' for Odd
    bytesize: int = 8
    timeout: float = 0.1 # Timeout en secondes pour les opérations Modbus
    mock: bool = False