"""
GPIO Hardware & 3-Way PWM Servo Physical Triage Controller
Supports physical Raspberry Pi 4 GPIO pins (RPi.GPIO / gpiozero) with automatic
fallback to an on-screen telemetry simulator when running in dry-run/benchmark mode.
"""

import time

try:
    import RPi.GPIO as GPIO
    RPI_HW_AVAILABLE = True
except ImportError:
    RPI_HW_AVAILABLE = False


class SmartTriageHardware:
    # BCM Pin Definitions for Raspberry Pi 40-Pin Header
    PIN_IR_SENSOR = 17      # Optical part-arrival trigger
    PIN_SERVO_PWM = 18      # Hardware PWM pin for 3-way diverter gate
    PIN_LED_PASS = 27       # Green LED (Nominal)
    PIN_LED_REWORK = 22     # Amber LED (Sector A / B Rework)
    PIN_LED_SCRAP = 23      # Red LED (Sector C Scrap)
    PIN_RELAY_INTERLOCK = 24  # Machine halt relay on 3 consecutive critical faults

    SECTOR_ANGLES = {
        "PASS": (0, "Main Conveyor -> Packaging Line"),
        "Minor": (30, "Sector A -> Cosmetic Buffing & Polish Bay"),
        "Moderate": (60, "Sector B -> Grinding, Welding & Rework Station"),
        "Critical": (90, "Sector C -> Structural Scrap Quarantine Bin"),
    }

    def __init__(self, simulate: bool = False):
        self.simulate = simulate or (not RPI_HW_AVAILABLE)
        self.current_angle = 0
        self.relay_tripped = False

        if not self.simulate:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.PIN_IR_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.PIN_SERVO_PWM, GPIO.OUT)
            GPIO.setup(self.PIN_LED_PASS, GPIO.OUT)
            GPIO.setup(self.PIN_LED_REWORK, GPIO.OUT)
            GPIO.setup(self.PIN_LED_SCRAP, GPIO.OUT)
            GPIO.setup(self.PIN_RELAY_INTERLOCK, GPIO.OUT)

            self.servo = GPIO.PWM(self.PIN_SERVO_PWM, 50)  # 50 Hz standard servo PWM
            self.servo.start(2.5)

    @staticmethod
    def _angle_to_duty_cycle(angle: int) -> float:
        return 2.5 + (float(angle) / 180.0) * 10.0

    def route_component(self, severity: str, consecutive_streak: int = 1) -> dict:
        angle, sector_desc = self.SECTOR_ANGLES.get(severity, self.SECTOR_ANGLES["PASS"])
        self.current_angle = angle
        self.relay_tripped = (severity == "Critical" and consecutive_streak >= 3)

        if not self.simulate:
            duty = self._angle_to_duty_cycle(angle)
            self.servo.ChangeDutyCycle(duty)
            GPIO.output(self.PIN_LED_PASS, severity == "PASS")
            GPIO.output(self.PIN_LED_REWORK, severity in ("Minor", "Moderate"))
            GPIO.output(self.PIN_LED_SCRAP, severity == "Critical")
            GPIO.output(self.PIN_RELAY_INTERLOCK, self.relay_tripped)
            time.sleep(0.05)

        return {
            "servo_angle_deg": angle,
            "triage_sector": sector_desc,
            "plc_interlock_halt": self.relay_tripped,
            "hw_mode": "SIMULATED_GPIO" if self.simulate else "RPI4_BCM_GPIO",
        }

    def cleanup(self):
        if not self.simulate:
            self.servo.stop()
            GPIO.cleanup()
