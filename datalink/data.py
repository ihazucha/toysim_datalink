import struct
import numpy as np
import pickle

from typing import Type, Any, Iterable, Tuple

# Serialization meta-classes
# -------------------------------------------------------------------------------------------------


class SerializableMeta(type):
    """Just to make sure every data class is correctly defined"""

    required_attributes: list[str] = []

    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)
        for attr in cls.required_attributes:
            if not hasattr(new_cls, attr):
                raise TypeError(f"Class {name} must define {attr} attribute")
        return new_cls


class Serializable(metaclass=SerializableMeta):
    @classmethod
    def from_bytes(cls, data: bytes):
        raise NotImplementedError()

    def to_bytes(self) -> bytes:
        raise NotImplementedError()

    def to_list(self) -> list:
        raise NotImplementedError()


class SerializablePickle(Serializable):
    @classmethod
    def from_bytes(cls, data: bytes):
        return pickle.loads(data)

    def to_bytes(self):
        return pickle.dumps(self)


class SerializablePrimitive(Serializable):
    required_attributes = ["SIZE", "FORMAT"]
    FORMAT: str = ""

    @classmethod
    def from_bytes(cls, data: bytes) -> "SerializablePrimitive":
        return cls(*struct.unpack(cls.FORMAT, data))

    def to_bytes(self) -> bytes:
        return struct.pack(self.__class__.FORMAT, *self.to_list())


class SerializableComplex(Serializable):
    required_attributes = ["SIZE", "COMPONENTS"]
    COMPONENTS: list[Type[Serializable]] = []

    @classmethod
    def from_bytes(cls, data: bytes):
        p = 0
        objects = []
        for c in cls.COMPONENTS:
            c_data = data[p : p + c.SIZE]
            o = c.from_bytes(c_data)
            objects.append(o)
            p += c.SIZE
        return cls(*objects)

    def to_bytes(self):
        return b"".join([c.to_bytes() for c in self.to_list()])


# General
# -------------------------------------------------------------------------------------------------


class Position(SerializablePrimitive):
    FORMAT = "=3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def __str__(self):
        return f"(x, y, z): ({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def to_list(self):
        return [self.x, self.y, self.z]


class Rotation(SerializablePrimitive):
    FORMAT = "=3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def __str__(self):
        return f"(r, p, y): ({self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f})"

    def to_list(self):
        return [self.roll, self.pitch, self.yaw]


class Pose(SerializableComplex):
    COMPONENTS = [Position, Rotation]
    SIZE = sum([c.SIZE for c in COMPONENTS])

    def __init__(self, position: Position, rotation: Rotation):
        self.position = position
        self.rotation = rotation

    def __str__(self):
        return f"{self.position} {self.rotation}"

    def to_list(self):
        return [self.position, self.rotation]


# Sensors
# -------------------------------------------------------------------------------------------------


class IMUData(SerializablePrimitive):
    FORMAT = "=Q6d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(
        self, timestamp: int, ax: float, ay: float, az: float, wr: float, wp: float, wy: float
    ):
        self.timestamp = timestamp
        self.ax, self.ay, self.az = ax, ay, az
        self.wr, self.wp, self.wy = wr, wp, wy

    def to_list(self):
        return [self.timestamp, self.ax, self.ay, self.az, self.wr, self.wp, self.wrs]


class IMU2Data(SerializablePickle):
    def __init__(self):
        self.timestamp: int
        self.accel_linear: Tuple[float, float, float] | None
        self.gyro: Tuple[float, float, float] | None
        self.mag: Tuple[float, float, float] | None
        self.rotation_euler_deg: Tuple[float, float, float] | None
        self.rotation_quaternion: Tuple[float, float, float, float] | None


class EncoderData(SerializablePrimitive):
    FORMAT = "=Qii"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, timestamp: int, position: int, magnitude: int):
        self.timestamp = timestamp
        self.position = position
        self.magnitude = magnitude

    def __str__(self):
        return f"tst: {self.timestamp}, pos: {self.position}, mag: {self.magnitude}"

    def to_list(self):
        return [self.timestamp, self.position, self.magnitude]


class RawImageData(Serializable):
    def __init__(self, timestamp: int, image_array: np.ndarray):
        self.timestamp = timestamp
        self.image_array = image_array

    @classmethod
    def from_bytes(cls, data: bytes):
        image_array = np.frombuffer(data[:-8], dtype=np.uint8)
        timestamp = struct.unpack("=Q", data[-8:])[0]
        return cls(timestamp, image_array)

    def to_bytes(self):
        image_array_bytes = self.image_array.tobytes()
        timestamp_bytes = struct.pack("=Q", self.timestamp)
        return image_array_bytes + timestamp_bytes

    def to_list(self):
        return [self.timestamp, self.image_array]

    @property
    def SIZE(self):
        return struct.calcsize("=Q") + self.image_array.nbytes


class JPGImageData(Serializable):
    def __init__(self, timestamp: int, jpg: bytes):
        self.timestamp = timestamp
        self.jpg = jpg

    @classmethod
    def from_bytes(cls, data: bytes):
        timestamp = struct.unpack("=Q", data[-8:])[0]
        return cls(timestamp, data[:-8])

    def to_bytes(self):
        timestamp_bytes = struct.pack("=Q", self.timestamp)
        return self.jpg + timestamp_bytes

    def to_list(self):
        return [self.timestamp, self.jpg]

    @property
    def SIZE(self):
        return struct.calcsize("=Q") + len(self.jpg)


# DEPRECATED
class SensorData(SerializableComplex):
    COMPONENTS = [IMUData, EncoderData, EncoderData, Pose]
    SIZE = sum([c.SIZE for c in COMPONENTS])

    def __init__(
        self,
        imu: IMUData,
        rleft_encoder: EncoderData,
        rright_encoder: EncoderData,
    ):
        self.imu = imu
        self.rleft_encoder = rleft_encoder
        self.rright_encoder = rright_encoder

    def to_list(self):
        return [self.imu, self.rleft_encoder, self.rright_encoder]


class SpeedometerData(SerializablePickle):
    def __init__(
        self, timestamp: int, dt: float, distance: float, speed: float, encoder_data: EncoderData
    ):
        self.timestamp = timestamp
        self.dt = dt
        self.distance = distance
        self.speed = speed
        self.encoder_data = encoder_data

    def __str__(self):
        return str(self.__dict__)


class SpeedometersData:
    def __init__(self, right_rear: SpeedometerData, left_rear: SpeedometerData):
        self.right_rear = right_rear
        self.left_rear = left_rear


class SensorFusionData(SerializablePickle):
    def __init__(
        self,
        timestamp: int,
        last_timestamp: int,
        dt: float,
        avg_speed: float,
        camera: JPGImageData,
        speedometer: Iterable[SpeedometerData],
        imu: Iterable[IMU2Data],
    ):
        self.timestamp = timestamp
        self.last_timestamp = last_timestamp
        self.dt = dt
        self.avg_speed = avg_speed
        self.camera = camera
        self.speedometer = speedometer
        self.imu = imu


# Common
# -------------------------------------------------------------------------------------------------


class ControlData(SerializablePrimitive):
    FORMAT = "=Q2d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, timestamp: int, speed: float, steering_angle: float):
        self.timestamp = timestamp
        self.speed = speed
        self.steering_angle = steering_angle

    def to_list(self):
        return [self.timestamp, self.speed, self.steering_angle]


# Real vehicle data
# -------------------------------------------------------------------------------------------------


class ActuatorsData(SerializablePrimitive):
    FORMAT = "=Q3d"
    SIZE = struct.calcsize(FORMAT)

    def __init__(
        self,
        timestamp: int,
        motor_power: float,
        steering_angle: float,
    ):
        self.timestamp = timestamp
        self.motor_power = motor_power
        self.steering_angle = steering_angle

    def to_list(self):
        return [self.timestamp, self.motor_power, self.steering_angle]


class RealData(Serializable):
    def __init__(
        self,
        timestamp: int,
        sensor_fusion: SensorFusionData,
        actuators: ActuatorsData,
        control: ControlData,
    ):
        self.timestamp = timestamp
        self.sensor_fusion = sensor_fusion
        self.actuators = actuators
        self.control = control

    @classmethod
    def from_bytes(cls, data: bytes) -> "RealData":
        return pickle.loads(data)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)


# Simulation
# -------------------------------------------------------------------------------------------------


class SimCameraData:
    FORMAT = "=4Q"
    W = 640
    H = 480
    RGB_IMAGE_SIZE = W * H * 3
    DEPTH_IMAGE_SIZE = W * H * 2
    SIZE = struct.calcsize(FORMAT) + RGB_IMAGE_SIZE + DEPTH_IMAGE_SIZE

    # TODO: rename rgb/depth_image to rgb/depth here and in UE
    def __init__(
        self,
        rgb_image: np.ndarray[Any, np.dtype[np.uint8]],
        depth_image: np.ndarray[Any, np.dtype[np.float16]],
        render_enqueued_unix_timestamp: int,
        render_finished_unix_timestamp: int,
        game_frame_number: int,
        render_frame_number: int,
    ):
        self.render_enqueued_unix_timestamp = render_enqueued_unix_timestamp
        self.render_finished_unix_timestamp = render_finished_unix_timestamp
        self.game_frame_number = game_frame_number
        self.render_frame_number = render_frame_number
        self.depth_image = depth_image
        self.rgb_image = rgb_image

    def to_bytes(self):
        # TODO: switch order of operations to avoid large array copy
        b = self.rgb_image.tobytes()
        b += self.depth_image.tobytes()
        b += struct.pack(
            SimCameraData.FORMAT,
            self.render_enqueued_unix_timestamp,
            self.render_finished_unix_timestamp,
            self.game_frame_number,
            self.render_frame_number,
        )
        return b

    def from_bytes(data: bytes) -> "SimCameraData":
        data_start = 0
        data_end = SimCameraData.RGB_IMAGE_SIZE
        rgb_image_array = np.frombuffer(data[:data_end], dtype=np.uint8)
        rgb_image_array = rgb_image_array.reshape((SimCameraData.H, SimCameraData.W, 3))

        data_start = data_end
        data_end += SimCameraData.DEPTH_IMAGE_SIZE
        depth_image_array = np.frombuffer(data[data_start:data_end], dtype=np.float16)
        depth_image_array = depth_image_array.reshape((SimCameraData.H, SimCameraData.W))

        data_start = data_end
        return SimCameraData(
            rgb_image_array,
            depth_image_array,
            *struct.unpack(SimCameraData.FORMAT, data[data_start:]),
        )


class SimVehicleData:
    FORMAT = "=2f"
    SIZE = struct.calcsize(FORMAT) + Pose.SIZE

    def __init__(self, speed: float, steering_angle: float, pose: Pose):
        self.speed = speed
        self.steering_angle = steering_angle
        self.pose = pose

    def __str__(self):
        return f"sped: {self.speed}, stra: {self.steering_angle}, pose: {self.pose}"

    def to_bytes(self):
        b = struct.pack(SimVehicleData.FORMAT, self.speed, self.steering_angle)
        b += self.pose.to_bytes()
        return b

    @staticmethod
    def from_bytes(data: bytes) -> "SimVehicleData":
        speed_and_steering_size = struct.calcsize(SimVehicleData.FORMAT)
        speed, steering_angle = struct.unpack(SimVehicleData.FORMAT, data[:speed_and_steering_size])
        pose = Pose.from_bytes(data[speed_and_steering_size:])
        return SimVehicleData(speed, steering_angle, pose)


class SimData:
    FORMAT = "=fQ"
    FORMAT_SIZE = struct.calcsize(FORMAT)
    SIZE = SimCameraData.SIZE + SimVehicleData.SIZE + FORMAT_SIZE

    def __init__(
        self, camera: SimCameraData, vehicle_data: SimVehicleData, dt: float, timestamp: int
    ):
        self.camera = camera
        self.vehicle = vehicle_data
        self.dt = dt
        self.timestamp = timestamp

    def to_bytes(self):
        b = self.camera.to_bytes()
        b += self.vehicle.to_bytes()
        b += struct.pack(SimData.FORMAT, self.dt, self.timestamp)
        return b

    @staticmethod
    def from_bytes(data: bytes) -> "SimData":
        data_memory_view = memoryview(data)
        data_start = 0
        data_end = SimCameraData.SIZE
        camera_data = SimCameraData.from_bytes(data_memory_view[:data_end])
        data_start = data_end
        data_end += SimVehicleData.SIZE
        vehicle_data = SimVehicleData.from_bytes(data_memory_view[data_start:data_end])
        data_start = data_end
        data_end += struct.calcsize(SimData.FORMAT)
        dt, timestamp = struct.unpack(SimData.FORMAT, data_memory_view[data_start:data_end])
        return SimData(camera_data, vehicle_data, dt, timestamp)


# Camera
# -------------------------------------------------------------------------------------------------


class ImageParams:
    def __init__(self, width: int, height: int, fov_deg: float):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg


# UI
# -------------------------------------------------------------------------------------------------


class PurePursuitConfig:
    def __init__(
        self,
        lookahead_factor: float,
        lookahead_dist_min: float,
        lookahead_dist_max: float,
        wheel_base: float,
        waypoint_shift: float,
    ):
        self.lookahead_factor = lookahead_factor
        self.lookahead_dist_min = lookahead_dist_min
        self.lookahead_dist_max = lookahead_dist_max
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift

    @classmethod
    def new_alamak(cls: "PurePursuitConfig") -> "PurePursuitConfig":
        return cls(
            lookahead_factor=2.2,
            lookahead_dist_min=0.6,
            lookahead_dist_max=2.0,
            wheel_base=0.185,
            waypoint_shift=0.180,
        )

    @classmethod
    def new_simulation(cls: "PurePursuitConfig") -> "PurePursuitConfig":
        return cls(
            lookahead_factor=2.2,
            lookahead_dist_min=500,
            lookahead_dist_max=2400,
            wheel_base=310,
            waypoint_shift=245,
        )


class PIDConfig:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd


class RoadmarksData(SerializablePickle):
    def __init__(self, roadmarks: np.ndarray, path: np.ndarray):
        self.roadmarks = roadmarks
        self.path = path


class ProcessedSimData(SerializablePickle):
    def __init__(
        self,
        begin_timestamp: int,
        control_data: ControlData,
        debug_image: np.ndarray,
        depth: np.ndarray,
        roadmarks_data: RoadmarksData,
        original: SimData,
    ):
        self.begin_timestamp = begin_timestamp
        self.control_data = control_data
        self.debug_image: np.ndarray = debug_image
        self.depth = depth
        self.original: SimData = original
        self.roadmarks_data = roadmarks_data


class ProcessedRealData(SerializablePickle):
    def __init__(
        self,
        begin_timestamp: int,
        control_data: ControlData,
        debug_image: np.ndarray,
        roadmarks_data: RoadmarksData,
        original: RealData,
    ):
        self.begin_timestamp = begin_timestamp
        self.control_data = control_data
        self.debug_image = debug_image
        self.roadmarks_data = roadmarks_data
        self.original = original
