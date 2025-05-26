# lerobot/common/robot_devices/robots/modbus_calibration.py
import json
import numpy as np
from pathlib import Path
import logging  # Import logging

# Make sure TorqueMode is accessible, adjust the path if necessary
from lerobot.common.robot_devices.motors.modbus_rtu_motor import ModbusRTUMotorsBus, TorqueMode


def run_modbus_rail_calibration(
    bus: ModbusRTUMotorsBus,
    motor_name_on_bus: str  # e.g., "axe_translation"
):
    """
    Calibrates a single Modbus motor on a given bus, typically a linear rail.
    Prompts the user to manually move the motor to its physical limits and enter the stroke length.
    Saves the calibration data (homing offset, steps per mm, limits) in a JSON file.
    """
    logging.info(f"\nStarting calibration for Modbus motor '{motor_name_on_bus}' on bus  ...")

    # Ensure torque is disabled for manual movement
    try:
        current_torque_arr = bus.read("Torque_Enable", motor_names_to_read=motor_name_on_bus)
        current_torque = current_torque_arr[0]  # Assuming read returns an array
    except Exception as e:
        logging.error(f"Unable to read torque state for {motor_name_on_bus}: {e}")
        logging.warning("Assuming torque is disabled and proceeding with caution.")
        current_torque = TorqueMode.DISABLED.value  # Safe assumption

    if current_torque != TorqueMode.DISABLED.value:
        logging.info(f"Disabling torque for motor '{motor_name_on_bus}'...")
        try:
            bus.write("Torque_Enable", TorqueMode.DISABLED.value, motor_names_to_write=motor_name_on_bus)
        except Exception as e:
            logging.error(f"Error while disabling torque for {motor_name_on_bus}: {e}")
            input("Failed to disable torque automatically. Please do it manually if possible, then press Enter.")

    input(f"Please manually move the motor carriage '{motor_name_on_bus}' to its physical MINIMUM position (e.g., fully left), then press Enter...")
    try:
        print(f"bus {bus}")
        print(f"motor_name_on_bus {motor_name_on_bus}")
        min_encoder_count_arr = bus.read("Present_Position")
        min_encoder_count = int(min_encoder_count_arr[0])
    except Exception as e:
        logging.error(f"Error reading minimum encoder position: {e}")
        min_encoder_count_str = input("Unable to read encoder position. Please manually enter the encoder value for the MINIMUM position: ")
        min_encoder_count = int(min_encoder_count_str)
    logging.info(f"Encoder steps at minimum position: {min_encoder_count}")

    input(f"Please manually move the motor carriage '{motor_name_on_bus}' to its physical MAXIMUM position (e.g., fully right), then press Enter...")
    try:
        max_encoder_count_arr = bus.read("Present_Position")
        max_encoder_count = int(max_encoder_count_arr[0])
    except Exception as e:
        logging.error(f"Error reading maximum encoder position: {e}")
        max_encoder_count_str = input("Unable to read encoder position. Please manually enter the encoder value for the MAXIMUM position: ")
        max_encoder_count = int(max_encoder_count_str)
    logging.info(f"Encoder steps at maximum position: {max_encoder_count}")

    if min_encoder_count >= max_encoder_count:
        logging.error(
            f"The minimum encoder count ({min_encoder_count}) is not less than the maximum count ({max_encoder_count}). "
            "Please check the movement direction or sensor."
        )
        # Allow user to correct
        min_encoder_count_str = input(f"Please re-enter the MINIMUM encoder value ({min_encoder_count}): ")
        min_encoder_count = int(min_encoder_count_str) if min_encoder_count_str else min_encoder_count
        max_encoder_count_str = input(f"Please re-enter the MAXIMUM encoder value ({max_encoder_count}): ")
        max_encoder_count = int(max_encoder_count_str) if max_encoder_count_str else max_encoder_count

        if min_encoder_count >= max_encoder_count:
            raise ValueError("Encoder values are still invalid after correction.")

    calibration_data = {
        "motor_names": motor_name_on_bus,
        "homing_offset_encoder_counts": min_encoder_count,
        "max_encoder_count": max_encoder_count,
    }
    logging.info(f"Calibration data for '{motor_name_on_bus}': {calibration_data}")

    input("Calibration setup complete. Press Enter to save and continue...")
    return calibration_data
