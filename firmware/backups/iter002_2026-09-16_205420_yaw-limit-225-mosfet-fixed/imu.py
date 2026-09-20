"""
GY-85 driver: ITG3205 gyro, ADXL345 accelerometer, HMC5883L magnetometer.

Why this is on the payload at all: everything the firmware has believed about
where the turret is has been DEAD RECKONED from commanded steps. That is only
true while nothing slips, and nothing here can tell when something does -- a
missed step is permanent and silent. A sensor on the moving body is the first
thing in this project that measures what the payload actually did.

What each part is worth here:

  gyro    Angular RATE, directly, at up to 1 kHz. This is the useful one. A
          velocity loop wants rate, so this closes the loop on the real thing
          instead of on an integral of what was commanded -- no differentiation
          of a noisy position estimate, no dependence on steps having landed.

  accel   Gravity, so absolute PITCH -- but only while the payload is not
          accelerating. Good for levelling and for setting a real datum;
          useless as feedback during a move.

  mag     Nothing usable. It is inches from two stepper motors and their field
          swamps the earth's. Absolute yaw is not available from this module;
          the camera is the yaw reference. Left readable for completeness and
          gated behind IMU_TRUST_MAG, which defaults False.
"""

import time

from machine import I2C, Pin

import config
import pinmap


class ImuError(Exception):
    usage = False


# --------------------------------------------------------------------------
#   registers
# --------------------------------------------------------------------------
_ITG_WHOAMI = 0x00
_ITG_SMPLRT_DIV = 0x15
_ITG_DLPF_FS = 0x16
_ITG_PWR_MGM = 0x3E
_ITG_GYRO_XOUT_H = 0x1D

_ADXL_DEVID = 0x00
_ADXL_POWER_CTL = 0x2D
_ADXL_DATA_FORMAT = 0x31
_ADXL_BW_RATE = 0x2C
_ADXL_DATAX0 = 0x32

_HMC_CRA = 0x00
_HMC_MODE = 0x02
_HMC_DATA = 0x03
_HMC_IDA = 0x0A


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def _s16le(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


class GY85:
    def __init__(self, port=None, freq=None):
        port = (port or config.IMU_PORT).upper()
        if port == "J5":
            sda, scl, iid = pinmap.I2C0_SDA, pinmap.I2C0_SCL, pinmap.I2C0_ID
        elif port == "J11":
            sda, scl, iid = pinmap.I2C_SDA, pinmap.I2C_SCL, pinmap.I2C_ID
        else:
            raise ImuError("IMU_PORT must be 'J5' or 'J11', not %r" % port)
        self.port = port
        self.bus = I2C(iid, sda=Pin(sda), scl=Pin(scl),
                       freq=freq or config.IMU_I2C_FREQ)
        self.have_gyro = False
        self.have_accel = False
        self.have_mag = False
        # Gyro bias, in raw LSB. Every MEMS gyro has an offset that drifts with
        # temperature; integrating an uncalibrated one walks away steadily.
        self.gyro_bias = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    def scan(self):
        try:
            return sorted(self.bus.scan())
        except OSError as e:
            raise ImuError("I2C%d on %s did not respond (%s). Check the "
                           "pull-ups -- this board has none."
                           % (0 if self.port == "J5" else 1, self.port, e))

    def _w(self, addr, reg, val):
        self.bus.writeto_mem(addr, reg, bytes([val]))

    def _r(self, addr, reg, n=1):
        return self.bus.readfrom_mem(addr, reg, n)

    # ------------------------------------------------------------------
    def begin(self):
        found = self.scan()
        g, a, m = (config.IMU_ADDR_GYRO, config.IMU_ADDR_ACCEL,
                   config.IMU_ADDR_MAG)

        if g in found:
            # PLL with X gyro as clock reference: the datasheet's recommended
            # source, more stable than the internal oscillator.
            self._w(g, _ITG_PWR_MGM, 0x01)
            time.sleep_ms(10)
            # DLPF 42 Hz, full scale +/-2000 deg/s (the only scale it has).
            self._w(g, _ITG_DLPF_FS, 0x18 | 0x03)
            self._w(g, _ITG_SMPLRT_DIV, 4)        # 1 kHz / (4+1) = 200 Hz
            self.have_gyro = True

        if a in found:
            self._w(a, _ADXL_POWER_CTL, 0x08)     # measure mode
            self._w(a, _ADXL_DATA_FORMAT, 0x0B)   # full res, +/-16 g
            self._w(a, _ADXL_BW_RATE, 0x0C)       # 400 Hz output
            self.have_accel = True

        if m in found:
            self._w(m, _HMC_CRA, 0x70)            # 8 averages, 15 Hz
            self._w(m, _HMC_MODE, 0x00)           # continuous
            self.have_mag = True

        if not (self.have_gyro or self.have_accel):
            raise ImuError(
                "no GY-85 on %s. Saw %s, expected gyro 0x%02X / accel 0x%02X."
                % (self.port, [hex(x) for x in found], g, a))
        return found

    # ------------------------------------------------------------------
    def gyro_raw(self):
        d = self._r(config.IMU_ADDR_GYRO, _ITG_GYRO_XOUT_H, 6)
        return (_s16(d[0], d[1]), _s16(d[2], d[3]), _s16(d[4], d[5]))

    def gyro_dps(self):
        """Angular rate in deg/s, bias removed."""
        x, y, z = self.gyro_raw()
        bx, by, bz = self.gyro_bias
        k = config.IMU_GYRO_LSB_PER_DPS
        return ((x - bx) / k, (y - by) / k, (z - bz) / k)

    def calibrate_gyro(self, samples=400, settle_ms=200):
        """
        Measure the zero-rate offset. THE PAYLOAD MUST BE STILL.

        Skipping this is the classic way to get a rate loop that slowly runs
        away: a constant offset of even a degree or two per second integrates
        into a steadily growing angle, and it looks exactly like mechanical
        drift.
        """
        time.sleep_ms(settle_ms)
        sx = sy = sz = 0
        for _ in range(samples):
            x, y, z = self.gyro_raw()
            sx += x
            sy += y
            sz += z
            time.sleep_ms(2)
        self.gyro_bias = (sx / samples, sy / samples, sz / samples)
        return self.gyro_bias

    # ------------------------------------------------------------------
    def accel_raw(self):
        d = self._r(config.IMU_ADDR_ACCEL, _ADXL_DATAX0, 6)
        return (_s16le(d[0], d[1]), _s16le(d[2], d[3]), _s16le(d[4], d[5]))

    def accel_g(self):
        x, y, z = self.accel_raw()
        k = 256.0                     # full-res mode is 3.9 mg/LSB
        return (x / k, y / k, z / k)

    def tilt_deg(self):
        """
        (pitch, roll) from gravity, in degrees.

        Valid ONLY when the payload is still. Any real acceleration adds to
        gravity and this reads it as tilt -- which is why it is a datum tool,
        not a feedback source.
        """
        import math
        x, y, z = self.accel_g()
        pitch = math.degrees(math.atan2(-x, math.sqrt(y * y + z * z)))
        roll = math.degrees(math.atan2(y, z))
        return (pitch, roll)

    def is_still(self, tol_dps=1.5):
        if not self.have_gyro:
            return False
        x, y, z = self.gyro_dps()
        return abs(x) < tol_dps and abs(y) < tol_dps and abs(z) < tol_dps

    # ------------------------------------------------------------------
    def mag_raw(self):
        d = self._r(config.IMU_ADDR_MAG, _HMC_DATA, 6)
        # HMC5883L order is X, Z, Y -- not X, Y, Z. Reading it in the obvious
        # order silently swaps two axes.
        return (_s16(d[0], d[1]), _s16(d[4], d[5]), _s16(d[2], d[3]))

    def describe(self):
        out = ["GY-85 on %s (I2C%d, %d Hz)"
               % (self.port, 0 if self.port == "J5" else 1,
                  config.IMU_I2C_FREQ)]
        out.append("  gyro  ITG3205  0x%02X  %s"
                   % (config.IMU_ADDR_GYRO,
                      "ok" if self.have_gyro else "NOT FOUND"))
        out.append("  accel ADXL345  0x%02X  %s"
                   % (config.IMU_ADDR_ACCEL,
                      "ok" if self.have_accel else "NOT FOUND"))
        out.append("  mag   HMC5883L 0x%02X  %s"
                   % (config.IMU_ADDR_MAG,
                      "ok" if self.have_mag else "not found"))
        if self.have_mag and not config.IMU_TRUST_MAG:
            out.append("        (readable, but not trusted: stepper fields)")
        out.append("  gyro bias %.1f, %.1f, %.1f LSB" % self.gyro_bias)
        return "\n".join(out)
