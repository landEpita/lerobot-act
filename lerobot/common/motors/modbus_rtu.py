# lerobot/common/motors/modbus_rtu.py
"""
Modbus-RTU single-axis bus (ex. rail linéaire NEMA17 + MKS SERVO42D).

• Compatible PyModbus 2.x **et** 3.x : wrapper interne qui gère `unit=` ↔ `slave=`
• API harmonisée avec les bus Dynamixel / Feetech :
  - Enum `TorqueMode` (ENABLED / DISABLED)
  - `read("Present_Position")`, `write("Goal_Position", …)`
  - Support `Torque_Enable`
• Conversion offset & pas-mm si `set_calibration()` appelé
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# PyModbus : import rétro-compatible 2.x / 3.x
# ---------------------------------------------------------------------------
try:  # PyModbus ≥ 3.0
    from pymodbus.client import ModbusSerialClient
except ImportError:  # PyModbus 2.5
    from pymodbus.client.sync import ModbusSerialClient  # type: ignore

from pymodbus.exceptions import ConnectionException, ModbusIOException
from pymodbus.payload import BinaryPayloadBuilder, Endian

# ---------------------------------------------------------------------------
# Registres spécifiques firmware MKS SERVO42D
# ---------------------------------------------------------------------------
REGISTER_ENCODER_READ_START = 0x30        # 3 × 16 bits : c_hi, c_lo, val(14b)
REGISTER_TORQUE_ENABLE      = 0xF3        # 0 = OFF / 1 = ON
REGISTER_GOAL_COMMAND_START = 0xF5        # (ACC, SPEED, ABS_AXIS)

ENC_MAX_CONST = 0x4000                    # 14 bits → 16384 counts/rev

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# Enum homogène avec Feetech / Dynamixel                                     #
# ---------------------------------------------------------------------------#
class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


# ---------------------------------------------------------------------------#
# Helper dataclass — utilisée dans les *configs* de robots                   #
# ---------------------------------------------------------------------------#
@dataclass
class ModbusRTUMotorsBusConfig:
    port: str                               # ex. "/dev/ttyUSB0"
    motors: Dict[str, tuple[int, str]]      # {"rail": (1, "NEMA17_MKS42D")}
    # Liaison série
    baudrate: int = 115_200
    parity: str = "N"                       # 'N','E','O'
    stopbits: int = 1
    bytesize: int = 8
    timeout: float = 0.1                    # s
    # Calibration par défaut
    microstep: int = 64
    steps_rev: int = 200                    # 1,8 ° → 200 pas
    acc_default: int = 100                  # registre F5
    speed_default: int = 300                # RPM
    # Registres alternatifs (rarement modifiés)
    present_pos_reg: int = REGISTER_ENCODER_READ_START
    goal_pos_reg: int = REGISTER_GOAL_COMMAND_START
    # Option mock (tests)
    mock: bool = False


# ---------------------------------------------------------------------------#
# Bus Modbus-RTU — aucune dépendance au « grand » MotorsBus ABC             #
# ---------------------------------------------------------------------------#
class ModbusRTUMotorsBus:
    """Bus Modbus-RTU multi-moteurs minimaliste (rail linéaire)."""

    # ------------------------------------------------------------------- #
    # Construction / init                                                 #
    # ------------------------------------------------------------------- #
    def __init__(self, cfg: ModbusRTUMotorsBusConfig):
        self.cfg      = cfg
        self.port     = cfg.port
        self.motors   = cfg.motors                # map name → (slave-id, model)
        self.mock     = cfg.mock

        # Client série
        self._client: ModbusSerialClient | None = None
        self.is_connected: bool = False
        self.logs: Dict[str, float] = {}

        # Pré-calculs utiles
        self.motor_names_list: List[str] = list(self.motors)
        self.motor_modbus_ids = {n: sid for n, (sid, _) in self.motors.items()}
        self.motor_models_map = {n: model for n, (_, model) in self.motors.items()}

        # Conversion µpas ↔ counts encodeur
        self.mu_step_rev = cfg.microstep * cfg.steps_rev         # µpas / 360°
        self.upas_to_counts = ENC_MAX_CONST / self.mu_step_rev   # facteur

        # Calibration (offset, max)
        self.calibration: Dict | None = None

    # ------------------------------------------------------------------- #
    # Connexion / déconnexion                                             #
    # ------------------------------------------------------------------- #
    def connect(self):
        if self.is_connected:
            raise ConnectionException(f"Modbus on {self.port} already connected")
        if self.mock:
            logger.info("[Modbus-Mock] Connected")
            self.is_connected = True
            return

        common = dict(
            port=self.port,
            baudrate=self.cfg.baudrate,
            parity=self.cfg.parity,
            stopbits=self.cfg.stopbits,
            bytesize=self.cfg.bytesize,
            timeout=self.cfg.timeout,
        )
        # Signature différente entre v2 et v3
        try:
            self._client = ModbusSerialClient(**common)           # PyModbus 3.x
        except TypeError:
            self._client = ModbusSerialClient(method="rtu", **common)  # PyModbus 2.x

        if not self._client.connect():                            # type: ignore[attr-defined]
            self._client = None
            raise ConnectionException(f"Unable to open {self.port}")
        self.is_connected = True
        logger.info("Modbus-RTU connected on %s", self.port)

    def disconnect(self):
        if not self.is_connected:
            return
        if self.mock:
            self.is_connected = False
            logger.info("[Modbus-Mock] Disconnected")
            return
        if self._client and self._client.is_socket_open():        # type: ignore[attr-defined]
            self._client.close()                                  # type: ignore[attr-defined]
        self._client = None
        self.is_connected = False
        logger.info("Modbus-RTU on %s closed", self.port)

    # ------------------------------------------------------------------- #
    # Calibration simple (offset + range)                                 #
    # ------------------------------------------------------------------- #
    def set_calibration(self, calib: Dict):
        """
        Attendu :
        {
            "motor_names": ["rail"],
            "homing_offset_encoder_counts": [0],
            "max_encoder_count":           [2000.0],
        }
        """
        if "rail_lineaire" in calib:
            self.calibration = calib["rail_lineaire"]
            logger.info("Calibration loaded for Modbus motors")
        else:
            raise ValueError("Calibration missing 'rail_lineaire' key")

    # ------------------------------------------------------------------- #
    # Helpers internes                                                    #
    # ------------------------------------------------------------------- #
    # Compat PyModbus 2.x / 3.x (unit ↔ slave)
    def _read_hreg(self, addr: int, slave: int):
        try:  # PyModbus ≥ 3.0  (count/slave mots-clés)
            return self._client.read_holding_registers(address=addr, count=1, slave=slave)
        except TypeError:  # PyModbus 2.x (count positionnel, unit=)
            return self._client.read_holding_registers(addr, 1, unit=slave)

    def _write_hreg(self, addr: int, value: int, slave: int):
        try:
            return self._client.write_register(address=addr, value=value, slave=slave)
        except TypeError:
            return self._client.write_register(addr, value, unit=slave)

    # —— lecture de l’encodeur (3 mots) ——
    def _read_encoder_raw(self, slave: int) -> int:
        if self.mock:
            return 0
        try:
            rr = self._client.read_input_registers(
                address=REGISTER_ENCODER_READ_START, count=3, slave=slave
            )
        except TypeError:  # signature 2.x
            rr = self._client.read_input_registers(
                REGISTER_ENCODER_READ_START, 3, unit=slave
            )

        if rr.isError():
            raise ModbusIOException(rr)

        c_hi, c_lo, val = rr.registers
        carry = (c_hi << 16) | c_lo
        if carry & 0x80000000:
            carry -= 0x100000000
        return carry * ENC_MAX_CONST + (val & 0x3FFF)
    
    def enable_torque(self, motor_names: Union[str, Sequence[str], None] = None) -> None:
        """Enable torque for specified motor(s)."""
        if motor_names is None:
            motor_names = self.motor_names_list
        self.write("Torque_Enable", TorqueMode.ENABLED, motor_names_to_write=motor_names)
        logger.info("Torque enabled for Modbus motor(s): %s", motor_names)

    def disable_torque(self, motor_names: Union[str, Sequence[str], None] = None) -> None:
        """Disable torque for specified motor(s)."""
        if motor_names is None:
            motor_names = self.motor_names_list
        self.write("Torque_Enable", TorqueMode.DISABLED, motor_names_to_write=motor_names)
        logger.info("Torque disabled for Modbus motor(s): %s", motor_names)


    # Offset / range
    def _get_offset_max(self, name: str):
        if not self.calibration:
            return 0, np.inf
        return (
            self.calibration["homing_offset_encoder_counts"],
            self.calibration["max_encoder_count"],
        )

    # µpas → counts driver (facultatif si on travaille direct en counts)
    def µpas_to_counts(self, ust: float) -> int:
        return int(round(ust * self.upas_to_counts))

    # ------------------------------------------------------------------- #
    # READ                                                                #
    # ------------------------------------------------------------------- #
    def read(
        self,
        item: str,
        motor_names_to_read: Union[str, Sequence[str], None] = None,
    ) -> np.ndarray:
        if not self.is_connected and not self.mock:
            raise RuntimeError("Modbus bus not connected")

        if motor_names_to_read is None:
            names = self.motor_names_list
        elif isinstance(motor_names_to_read, str):
            names = [motor_names_to_read]
        else:
            names = list(motor_names_to_read)

        raws: List[int] = []
        for n in names:
            sid = self.motor_modbus_ids[n]
            if item == "Present_Position":
                raw = self._read_encoder_raw(sid) if not self.mock else 0
            elif item == "Torque_Enable":
                if self.mock:
                    raw = TorqueMode.ENABLED.value
                else:
                    rr = self._read_hreg(REGISTER_TORQUE_ENABLE, sid)
                    if rr.isError():                                   # type: ignore[attr-defined]
                        raise ModbusIOException(rr)
                    raw = rr.registers[0]                              # type: ignore[attr-defined]
            else:
                logger.warning("Read %s not implemented for Modbus", item)
                raw = 0
            raws.append(raw)

        arr = np.asarray(raws, dtype=np.int64)
        if item == "Present_Position":
            if self.calibration:
                phys = np.zeros_like(arr, dtype=np.float32)
                off, _max = self._get_offset_max(names[0])
                phys[:] = arr - off
                return phys
            return arr.astype(np.float32)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------- #
    # WRITE                                                               #
    # ------------------------------------------------------------------- #
    def _build_f5_payload(self, counts: int, acc: int, speed: int):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
        builder.add_16bit_uint(acc)
        builder.add_16bit_uint(speed)
        builder.add_32bit_int(counts)
        return builder.to_registers()

    def write(
        self,
        item: str,
        values: Union[int, float, TorqueMode, np.ndarray, Dict[str, float]],
        *,
        motor_names_to_write: Union[str, Sequence[str], None] = None,
    ) -> None:
        if not self.is_connected and not self.mock:
            raise RuntimeError("Modbus bus not connected")

        # Normaliser <name,value>
        if isinstance(values, dict):
            kv_pairs = values.items()
        else:
            # scalaire ou vecteur → broadcast
            if motor_names_to_write is None:
                names = self.motor_names_list
            elif isinstance(motor_names_to_write, str):
                names = [motor_names_to_write]
            else:
                names = list(motor_names_to_write)
            arr = np.array(values).flatten()
            if arr.size == 1 and len(names) > 1:
                arr = np.repeat(arr, len(names))
            if arr.size != len(names):
                raise ValueError("Mismatch motor names / values")
            kv_pairs = zip(names, arr, strict=True)

        for name, val in kv_pairs:
            sid = self.motor_modbus_ids[name]
            if self.mock:
                logger.debug("[Modbus-Mock] write %s %s → %s", item, val, name)
                continue

            if item == "Torque_Enable":
                if isinstance(val, TorqueMode):
                    val = val.value
                rq = self._write_hreg(REGISTER_TORQUE_ENABLE, int(val), sid)
                if rq.isError():                                         # type: ignore[attr-defined]
                    raise ModbusIOException(rq)

            elif item == "Goal_Position":
                # -> encoder counts (offset + clamp)
                counts = int(val)
                if self.calibration:
                    off, max_cnt = self._get_offset_max(name)
                    counts = int(np.clip(counts + off, off, max_cnt))
                regs = self._build_f5_payload(
                    counts, self.cfg.acc_default, self.cfg.speed_default
                )
                try:  # PyModbus 3.x : write_registers(slave=)
                    rq = self._client.write_registers(  # type: ignore[attr-defined]
                        REGISTER_GOAL_COMMAND_START, regs, slave=sid
                    )
                except TypeError:  # PyModbus 2.x : unit=
                    rq = self._client.write_registers(  # type: ignore[attr-defined]
                        REGISTER_GOAL_COMMAND_START, regs, unit=sid
                    )
                if rq.isError():                                      # type: ignore[attr-defined]
                    raise ModbusIOException(rq)
            else:
                logger.warning("Write %s not implemented for Modbus", item)

    # ------------------------------------------------------------------- #
    # Misc. helpers                                                       #
    # ------------------------------------------------------------------- #
    @property
    def motor_names(self) -> List[str]:
        return self.motor_names_list

    @property
    def motor_models(self) -> List[str]:
        return [self.motor_models_map[n] for n in self.motor_names_list]

    # Nothing special to setup – ID is fixed in driver
    def setup_motor(self, motor_name: str) -> None:  # noqa: D401
        print(f"(Modbus) nothing to setup for {motor_name}")

    # Destructor safety
    def __del__(self):
        try:
            if self.is_connected:
                self.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.error("ModbusRTUMotorsBus.__del__ error: %s", exc)
