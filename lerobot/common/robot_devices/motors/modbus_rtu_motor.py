"""Modbus RTU motor bus compatible avec l’API LeRobot.

Cette implémentation s’aligne sur les bus Dynamixel et Feetech :
- même énumération `TorqueMode` (ENABLED / DISABLED)
- méthodes `read` et `write` acceptant/renvoyant `Torque_Enable`
- conversion automatique Enum ↔︎ int dans `write`
- aucun changement nécessaire côté `ManipulatorRobot`

Les registres sont basés sur le firmware MKS SERVO42D (axe linéaire NEMA‑17).
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from typing import List, Sequence, Union

import numpy as np
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException, ConnectionException
from pymodbus.payload import BinaryPayloadBuilder, Endian

from lerobot.common.robot_devices.motors.configs import ModbusRTUMotorsBusConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc

# -----------------------------------------------------------------------------
# Registres spécifiques au firmware MKS SERVO42D
# -----------------------------------------------------------------------------
REGISTER_ENCODER_READ_START = 0x30  # 3 mots : c_hi, c_lo, val (14 bits)
REGISTER_TORQUE_ENABLE = 0xF3       # 0 = OFF, 1 = ON
REGISTER_GOAL_COMMAND_START = 0xF5  # F5h : ACC, SPEED, ABS_AXIS (32 bits)

ENC_MAX_CONST = 0x4000  # 14 bits -> 16384 pas encodeur/cycle

# -----------------------------------------------------------------------------
# Harmonisation : Enum identique à Dynamixel / Feetech
# -----------------------------------------------------------------------------
class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class ModbusRTUMotorsBus:
    """Bus Modbus‑RTU gérant un ou plusieurs moteurs (ex. rail linéaire)."""

    # ---------------------------------------------------------------------
    # Initialisation / connexion
    # ---------------------------------------------------------------------
    def __init__(self, config: ModbusRTUMotorsBusConfig):
        self.config = config
        self.port = config.port
        self.motors = config.motors  # {"axe_translation": (201, "NEMA17_MKS42D"), ...}
        self.mock = config.mock

        self.client: ModbusSerialClient | None = None
        self.is_connected = False
        self.calibration: dict | None = None
        self.logs: dict = {}

        # Pré‑calculs pratiques
        self.motor_names_list: List[str] = list(self.motors.keys())
        self.motor_modbus_ids = {name: info[0] for name, info in self.motors.items()}
        self.motor_models_map = {name: info[1] for name, info in self.motors.items()}

        # Paramètres moteur (valeurs par défaut, surchargeables via config)
        self.microstep = getattr(config, "microstep", 64)
        self.steps_rev = getattr(config, "steps_rev", 200)  # 1,8° = 200 pas
        self.acc_default = getattr(config, "acc_default", 100)  # 0‑255
        self.speed_default = getattr(config, "speed_default", 300)  # RPM

        self.mu_step_rev = self.microstep * self.steps_rev  # micropas/360°
        self.upas_to_counts = ENC_MAX_CONST / self.mu_step_rev 

    # ------------------------ connexion / déconnexion --------------------
    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"ModbusRTUMotorsBus({self.port}) already connected"
            )
        if self.mock:
            logging.info("[Modbus‑Mock] Connected")
            self.is_connected = True
            return

        try:
            self.client = ModbusSerialClient(
                port=self.port,
                baudrate=self.config.baudrate,
                stopbits=self.config.stopbits,
                parity=self.config.parity,
                bytesize=self.config.bytesize,
                timeout=self.config.timeout,
            )
            if not self.client.connect():
                raise ConnectionException(f"Unable to open {self.port}")
            self.is_connected = True
            logging.info(f"ModbusRTUMotorsBus({self.port}) connected")
        except Exception as exc:
            self.client = None
            raise ConnectionException(f"Modbus connect failed on {self.port}: {exc}")

    def disconnect(self):
        if not self.is_connected and not self.mock:
            return
        if self.mock:
            self.is_connected = False
            logging.info("[Modbus‑Mock] Disconnected")
            return
        if self.client and self.client.is_socket_open():
            self.client.close()
        self.is_connected = False
        self.client = None
        logging.info(f"ModbusRTUMotorsBus({self.port}) disconnected")

    # ---------------------------------------------------------------------
    # Propriétés utilitaires
    # ---------------------------------------------------------------------
    @property
    def motor_names(self) -> List[str]:
        return self.motor_names_list

    @property
    def motor_models(self) -> List[str]:
        return [self.motor_models_map[n] for n in self.motor_names_list]

    # ---------------------------------------------------------------------
    # Calibration simple (offset + pas/mm)
    # ---------------------------------------------------------------------
    def set_calibration(self, calibration: dict):
        """Exemple :
        {
            "motor_names": ["axe_translation"],
            "homing_offset_encoder_counts": [0],
            "max_encoder_count": [2000.0],
        }
        """
        if "rail_lineaire" in calibration:
            self.calibration = calibration["rail_lineaire"]
            logging.info("Calibration loaded for Modbus motors")
        else:
            raise ValueError("Invalid calibration data for Modbus motors")

    # ------------------- helpers calibration interne ---------------------
    def _get_motor_calib_params(self, motor_name: str):
        if not self.calibration:
            return 0, np.inf
        offset = self.calibration["homing_offset_encoder_counts"]
        he = self.calibration["max_encoder_count"]
        return offset, he
    
    def µpas_to_counts(self, ust): 
        return int(round(ust * self.upas_to_counts))

    # ---------------------------------------------------------------------
    # Encoders
    # ---------------------------------------------------------------------
    def _read_encoder_raw(self, slave_id: int) -> int:
        if self.mock:
            return 0
        rr = self.client.read_input_registers(
            address=REGISTER_ENCODER_READ_START, count=3, slave=slave_id
        )
        if rr.isError():
            raise ModbusIOException(f"Encoder read error (slave {slave_id}): {rr}")
        c_hi, c_lo, val = rr.registers
        carry = (c_hi << 16) | c_lo
        if carry & 0x80000000:
            carry -= 0x100000000
        return carry * ENC_MAX_CONST + (val & 0x3FFF)

    # ---------------------------------------------------------------------
    # Calibration conversions
    # ---------------------------------------------------------------------
    def apply_calibration(self, raw_counts: np.ndarray, names: Sequence[str]):
        if not self.calibration:
            return raw_counts.astype(np.float32)
        phys = np.zeros_like(raw_counts, dtype=np.float32)
        for i, n in enumerate(names):
            off, _ = self._get_motor_calib_params(n)
            # print(f"raw_counts {raw_counts[i]} off {off}")
            phys[i] = (raw_counts[i] - off)
            # print(f"phys {phys[i]}")
        return phys

    def revert_calibration(self, phys_vals: np.ndarray, names: Sequence[str]):
        """Ajoute seulement l’offset. Vérifie que le résultat est dans [offset, max]."""
        if not self.calibration:
            return phys_vals.astype(np.int64)

        enc = np.zeros_like(phys_vals, dtype=np.int64)
        for i, n in enumerate(names):
            offset, enc_max = self._get_motor_calib_params(n)

            val = phys_vals[i]
            # Ajout de l’offset (donc conversion en encoder count)
            val = min(enc_max, val)
            val = max(offset, val)

            target_enc = int(round(val + offset))



            enc[i] = target_enc
        return enc

    # ---------------------------------------------------------------------
    # READ
    # ---------------------------------------------------------------------
    def read(
        self, data_name: str, motor_names_to_read: Union[str, Sequence[str], None] = None
    ) -> np.ndarray:
        if not self.is_connected and not self.mock:
            raise RobotDeviceNotConnectedError("Modbus bus not connected")

        if motor_names_to_read is None:
            motor_names_to_read = self.motor_names_list
        elif isinstance(motor_names_to_read, str):
            motor_names_to_read = [motor_names_to_read]

        raw_vals: List[int] = []
        for n in motor_names_to_read:
            sid = self.motor_modbus_ids[n]
            if data_name == "Present_Position":
                raw = self._read_encoder_raw(sid) if not self.mock else 0
            elif data_name == "Torque_Enable":
                if self.mock:
                    raw = int(TorqueMode.ENABLED.value)
                else:
                    rr = self.client.read_holding_registers(
                        REGISTER_TORQUE_ENABLE, count=1, slave=sid
                    )
                    if rr.isError():
                        raise ModbusIOException(rr)
                    raw = rr.registers[0]
            else:
                logging.warning(f"Read {data_name} not implemented for Modbus (motor {n})")
                raw = 0
            raw_vals.append(raw)
        raw_arr = np.array(raw_vals, dtype=np.int64)
        if data_name == "Present_Position":
            out = self.apply_calibration(raw_arr, motor_names_to_read)
        else:
            out = raw_arr.astype(np.float32)

        self.logs[f"read_{data_name}_dt_s"] = 0  # simplifié
        self.logs[f"read_{data_name}_timestamp_utc"] = capture_timestamp_utc()
        return out

    # ---------------------------------------------------------------------
    # WRITE
    # ---------------------------------------------------------------------
    def _build_f5_payload(self, abs_driver_steps: int, acc: int, speed: int):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        builder.add_16bit_uint(acc)
        builder.add_16bit_uint(speed)
        builder.add_32bit_int(abs_driver_steps)
        return builder.to_registers()

    def write(
        self,
        data_name: str,
        values_to_write: Union[int, float, TorqueMode, np.ndarray],
        motor_names_to_write: Union[str, Sequence[str], None] = None,
    ) -> None:
        if not self.is_connected and not self.mock:
            raise RobotDeviceNotConnectedError("Modbus bus not connected")

        if motor_names_to_write is None:
            motor_names_to_write = self.motor_names_list
        elif isinstance(motor_names_to_write, str):
            motor_names_to_write = [motor_names_to_write]

        # Normaliser les valeurs ------------------------------------------------
        if isinstance(values_to_write, TorqueMode):
            values_np = np.asarray([values_to_write.value] * len(motor_names_to_write))
        elif isinstance(values_to_write, (int, float)):
            values_np = np.asarray([values_to_write] * len(motor_names_to_write))
        else:
            values_np = np.asarray(values_to_write)
            if values_np.size == 1 and len(motor_names_to_write) > 1:
                values_np = np.repeat(values_np, len(motor_names_to_write))
        if values_np.size != len(motor_names_to_write):
            raise ValueError("Mismatch values/motors in Modbus.write")


        # Boucle écriture -------------------------------------------------------
        for val, name in zip(values_np, motor_names_to_write, strict=True):
            sid = self.motor_modbus_ids[name]
            if self.mock:
                logging.debug(f"[Modbus‑Mock] write {data_name} {val} → {name}")
                continue

            try:
                if data_name == "Torque_Enable":
                    rq = self.client.write_register(
                        address=REGISTER_TORQUE_ENABLE, value=int(val), slave=sid
                    )
                    if rq.isError():
                        raise ModbusIOException(rq)

                elif data_name == "Goal_Position":
                    enc_target = self.revert_calibration(np.asarray([val]), [name])[0]
                    # print(f"enc_target {enc_target}")
                    # driver_steps = int(self.µpas_to_counts(enc_target))
                    # print(f"driver_steps {driver_steps}")
                    regs = self._build_f5_payload(enc_target, self.acc_default, self.speed_default)
                    rq = self.client.write_registers(
                        address=REGISTER_GOAL_COMMAND_START, values=regs, slave=sid
                    )
                    if rq.isError():
                        raise ModbusIOException(rq)

                else:
                    logging.warning(f"Write {data_name} not implemented for Modbus (motor {name})")
            except Exception as exc:
                logging.error(f"Modbus write error ({data_name}, motor {name}): {exc}")

        self.logs[f"write_{data_name}_timestamp_utc"] = capture_timestamp_utc()

    # ---------------------------------------------------------------------
    # Destructor -----------------------------------------------------------
    # ---------------------------------------------------------------------
    def __del__(self):
        try:
            if self.is_connected:
                self.disconnect()
        except Exception as exc:
            logging.critical(f"ModbusRTUMotorsBus.__del__ error: {exc}")
