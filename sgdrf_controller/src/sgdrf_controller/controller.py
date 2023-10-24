from enum import Enum
from transitions import EventData
from transitions.extensions import LockedMachine
from typing import Callable
from threading import Event, RLock
import inspect
import rospy
import filterpy.kalman
import numpy as np
from ifcbclient.protocol import parse_response as parse_ifcb_msg
from ifcb.instrumentation import parse_marker as parse_ifcb_marker

from ds_core_msgs.msg import RawData

from ifcb.srv import RunRoutine
from phyto_arm.msg import ConductorState, ConductorStates

from gps_common.msg import GPSFix
import utm

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from pathlib import Path

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from std_msgs.msg import String

CONFIG_FILENAME = (
    Path(__file__).parents[4] / "share" / "sgdrf_controller" / "state_machine.yaml"
)

RADIANS_PER_DEGREE = np.pi / 180


def log_ros(log_func: Callable = rospy.logdebug) -> Callable:
    def wrapper(func: Callable[[EventData], None]) -> Callable[[EventData], None]:
        def inner(_, event: EventData):
            if event.event:
                log_str = f"triggered event '{event.event.name}'"
                if event.transition:
                    log_str += f"from state '{event.transition.source}' to state '{event.transition.dest}'"
                else:
                    log_str += f"in state '{event.state.name}'"
                log_func(log_str)
            return func(_, event)

        return inner

    return wrapper


class Controller:
    def __init__(self) -> None:
        self.lock = RLock()
        with open(CONFIG_FILENAME, "r") as f:
            config = load(f, Loader=Loader)
        self.all_states = config["states"]
        self.all_transitions = config["transitions"]
        nonautonomous_triggers = {"to_error", "begin_idling", "turn_on", "turn_off"}
        self.autonomous_triggers = [
            t["trigger"]
            for t in self.all_transitions
            if t["trigger"] not in nonautonomous_triggers
        ]
        self.machine = LockedMachine(
            model=self,
            states=self.all_states,
            initial="controller_off",
            auto_transitions=False,
            send_event=True,
            machine_context=self.lock,
            transitions=self.all_transitions,
        )
        self.ifcb_run_routine = None
        self.gps_fix_sub = None
        self.on_ifcb_msg_sub = None
        self.set_state_pub = None
        self.last_cart_debub_time = None
        self.last_bead_time = None
        self.ifcb_is_idle = Event()
        self.manually_idled = Event()
        self.loop_timer = None
        self.lat = 0.0
        self.lon = 0.0
        self.max_lat = rospy.get_param("max_lat", 180)
        self.min_lat = rospy.get_param("min_lat", -180)

        # self.filter = filterpy.kalman.UnscentedKalmanFilter(
        #     dim_x=6,
        #     dim_z=4,
        #     dt=1,
        #     hx=self.kalman_hx,
        #     fx=self.kalman_fx,
        #     points=filterpy.kalman.MerweScaledSigmaPoints(
        #         4, alpha=0.1, beta=2.0, kappa=-1
        #     ),
        #     z_mean_fn=self.kalman_z_mean_fn,
        #     residual_z=self.kalman_residual_z,
        # )
        # self.filter_initialized = False

        self.bead_interval = rospy.Duration(60 * rospy.get_param("bead_interval", 120))
        self.cart_debub_interval = rospy.Duration(
            60 * rospy.get_param("cartridge_debubble_interval", 143)
        )

        self.on_enter_sample_standard = self.run_routine_callback("runsample")
        self.on_enter_debubble_standard = self.run_routine_callback("debubble")
        self.on_enter_debubble_adaptive = self.run_routine_callback("debubble")
        self.on_enter_beads_sequence_debubble = self.run_routine_callback("debubble")
        self.on_enter_beads_sequence_beads = self.run_routine_callback("beads")
        self.on_enter_beads_sequence_biocide = self.run_routine_callback("biocide")
        self.on_enter_beads_sequence_bleach = self.run_routine_callback("bleach")
        self.on_enter_cartridge_debubble = self.run_routine_callback(
            "cartridgedebubble"
        )
        self.state_diag_pub = None

        self.turn_off_service = None
        self.turn_on_service = None
        self.idle_service = None
        self.stop_idling_service = None
        self.reset_clean_times_service = None

    @staticmethod
    def kalman_hx(x: np.array) -> np.array:
        z = np.zeros_like(x, shape=(4,))
        z[0] = x[0]
        z[1] = x[3]
        z[2] = (90 - np.arctan2(x[3], x[0]) / RADIANS_PER_DEGREE) % 360
        z[3] = np.sqrt(x[1] ** 2 + x[4] ** 2)

    @staticmethod
    def kalman_fx(x: np.array, dt: float) -> np.array:
        m = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        return np.dot(m, x)

    @staticmethod
    def kalman_z_mean_fn(sigmas: np.array, wm: np.array) -> np.array:
        z = np.zeros(4)
        sum_sin, sum_cos = 0.0, 0.0

        for i in range(len(sigmas)):
            s = sigmas[i]
            z[0] += s[0] * wm[i]
            z[1] += s[1] * wm[i]
            z[3] += s[3] * wm[i]
            sum_sin += np.sin(s[2] * RADIANS_PER_DEGREE) * wm[i]
            sum_cos += np.cos(s[2] * RADIANS_PER_DEGREE) * wm[i]
        z[2] = np.arctan2(sum_sin, sum_cos)
        return z

    @staticmethod
    def kalman_residual_z(x: np.array, y: np.array) -> np.array:
        ret = x - y
        ret[2] = (x[2] - y[2]) % 360
        return ret

    @staticmethod
    def kalman_residual_z(x: np.array, y: np.array) -> np.array:
        ret = x - y
        ret[2] = (x[2] - y[2]) % 360
        return ret

    def initialize_filter(self, msg: GPSFix):
        self.filter.x = self.msg_to_x(msg)
        self.filter.Q = filterpy.common.Q_discrete_white_noise(
            dim=2, dt=1.0, var=1.0, block_size=2
        )
        self.filter.predict()
        self.filter_initialized = True

    def gpsfix_callback(self, msg: GPSFix):
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.track = msg.track
        self.speed = msg.speed
        # if self.filter_initialized:
        #     self.filter.update(self.msg_to_z(msg))
        # else:
        #     self.initialize_filter(msg)

    def setup_pub_sub_srv(self):
        self.ifcb_run_routine = rospy.ServiceProxy("/ifcb/routine", RunRoutine)
        self.gps_fix_sub = rospy.Subscriber(
            rospy.get_param("/sgdrf_controller/gps_topic_name", "/gps/extended_fix"),
            GPSFix,
            self.gpsfix_callback,
        )
        self.on_ifcb_msg_sub = rospy.Subscriber("/ifcb/in", RawData, self.on_ifcb_msg)
        self.set_state_pub = rospy.Publisher(
            "~state", ConductorState, queue_size=1, latch=True
        )
        self.turn_off_service = rospy.Service(
            "/sgdrf/turn_off_controller", Trigger, self.turn_off_service_provider
        )
        self.turn_on_service = rospy.Service(
            "/sgdrf/turn_on_controller", Trigger, self.turn_on_service_provider
        )
        self.idle_service = rospy.Service(
            "/sgdrf/idle_controller", Trigger, self.idle_service_provider
        )
        self.stop_idling_service = rospy.Service(
            "/sgdrf/stop_idling", Trigger, self.stop_idling_service_provider
        )
        self.reset_clean_times_service = rospy.Service(
            "/sgdrf/reset_clean_times", Trigger, self.reset_clean_times_service_provider
        )

    def set_state(self, s: ConductorStates):
        m = ConductorState()
        m.header.stamp = rospy.Time.now()
        m.state = s
        self.set_state_pub.publish(m)

    def on_ifcb_msg(self, msg: RawData):
        # Parse the message and see if it is a marker
        marker = None
        parsed = parse_ifcb_msg(msg.data.decode())
        if len(parsed) == 2 and parsed[0] == "reportevent":
            marker = parse_ifcb_marker(parsed[1])

        # For routines sent with 'interactive:start', the IFCB will tell us when the
        # routine has started and finished. This is the most reliable way to detect
        # this, rather than using the markers.
        #
        # XXX
        # The ifcb_is_idle Event should be clear()'d _before_ sending a routine, to
        # avoid a race condition where a thread sends a routine and then immediately
        # awaits the Event before we have been told the routine started.
        if parsed == ["valuechanged", "interactive", "stopped"]:
            rospy.loginfo("IFCB routine not running")
            self.ifcb_is_idle.set()
            return

        # Remaining behaviors depend on having a parsed marker
        if marker is None:
            return

    @staticmethod
    def msg_to_z(msg: GPSFix) -> np.array:
        lat = msg.latitude
        lon = msg.longitude
        track = msg.track
        speed = msg.speed

        easting, northing, _, _ = utm.from_latlon(lat, lon)

        return np.array([easting, northing, track, speed])

    @staticmethod
    def msg_to_x(msg: GPSFix) -> np.array:
        lat = msg.latitude
        lon = msg.longitude
        track = msg.track
        speed = msg.speed
        easting, northing, _, _ = utm.from_latlon(lat, lon)
        vy = speed * np.cos(track * RADIANS_PER_DEGREE)
        vx = np.sqrt(speed**2 - vy**2)
        # No acceleration measurements, so these are set to 0
        return np.array([easting, vx, 0, northing, vy, 0.0])

    def init_node(self):
        rospy.init_node("sgdrf_conductor", anonymous=True, log_level=rospy.DEBUG)

        self.machine_setup_state_callbacks()
        self.setup_pub_sub_srv()

        # Initialize service proxy for sending routines to the IFCB
        self.ifcb_run_routine.wait_for_service()

        # Set a fake timestamp for having run beads and catridge debubble, so that
        # we don't run it every startup and potentially waste time or bead supply.
        self.last_cart_debub_time = rospy.Time.now()
        self.last_bead_time = rospy.Time.now()
        self.loop_timer = rospy.Timer(rospy.Duration(1.0), self.loop)

    def machine_setup_state_callbacks(self):
        for state in self.machine.states:
            for fname in ["on_enter", "on_exit"]:
                self.machine.__getattr__(f"{fname}_{state}")(
                    f"{fname}_{state}"
                    if hasattr(self, f"{fname}_{state}")
                    else self.empty_callback
                )

    @log_ros()
    def on_enter_controller_off(self, _: EventData):
        rospy.loginfo("Turning controller off.")

    @log_ros()
    def on_exit_controller_off(self, _: EventData):
        rospy.loginfo("Turning controller on.")

    @log_ros()
    def on_enter_idle(self, _: EventData):
        rospy.loginfo("Idling.")

    @log_ros()
    def on_enter_error(self, event: EventData):
        error_info = event.kwargs.get("error_info", "none provided")
        rospy.logerr(f"Entered error state. Error: {error_info}")

    @log_ros()
    def on_exit_error(self, _: EventData):
        rospy.loginfo("Clearing error state.")

    def run_routine_callback(self, routine: str):
        @log_ros()
        def callback(self, _: EventData):
            self.ifcb_is_idle.clear()
            self.ifcb_run_routine(routine=routine, instrument=True)

        return callback

    @log_ros()
    def on_enter_sample_adaptive(self, _: EventData):
        # TODO
        self.ifcb_is_idle.clear()
        self.ifcb_run_routine(routine="runsample", instrument=True)
        pass

    @log_ros()
    def on_enter_at_top(self, _: EventData):
        pass

    @log_ros()
    def on_enter_turning_to_go_south(self, _: EventData):
        pass

    @log_ros()
    def on_enter_ready_to_go_south(self, _: EventData):
        pass

    @log_ros()
    def on_enter_at_bottom(self, _: EventData):
        pass

    @log_ros()
    def on_enter_turning_to_go_north(self, _: EventData):
        pass

    @log_ros()
    def on_enter_ready_to_go_north(self, _: EventData):
        pass

    @log_ros(rospy.logerr)
    def on_exception(self, _: EventData):
        pass

    @log_ros()
    def is_ifcb_idle_or_can_acquire_better_sample(self, event: EventData) -> bool:
        return self.is_ifcb_idle(event) or self.can_acquire_better_sample(event)

    @log_ros()
    def can_acquire_better_sample(self, _: EventData) -> bool:
        # TODO
        return False

    @log_ros()
    def is_ifcb_idle(self, _: EventData) -> bool:
        return self.ifcb_is_idle.is_set()

    @log_ros()
    def is_at_top(self, _: EventData) -> bool:
        return np.isclose(self.lat, self.max_lat)

    @log_ros()
    def is_at_bottom(self, _: EventData) -> bool:
        return np.isclose(self.lat, self.min_lat)

    @log_ros()
    def is_stopped(self, _: EventData):
        return self.speed < 0.1

    @log_ros()
    def is_moving(self, _: EventData):
        return self.speed > 1.0

    @log_ros()
    def is_going_north(self, _: EventData) -> bool:
        return np.isclose(self.track, 0) or np.isclose(self.track, 360)

    @log_ros()
    def is_going_south(self, _: EventData) -> bool:
        return np.isclose(self.track, 180)

    @log_ros()
    def is_it_time_to_run_beads(self, _: EventData) -> bool:
        run_beads = not np.isclose(self.bead_interval.to_sec(), 0.0)
        enough_time_passed = (
            rospy.Time.now() - self.last_bead_time
        ) > self.bead_interval
        return run_beads and enough_time_passed

    @log_ros()
    def is_it_time_to_run_cartridge_debubble(self, _: EventData) -> bool:
        run_cart_debub = not np.isclose(self.cart_debub_interval.to_sec(), 0.0)
        enough_time_passed = (
            rospy.Time.now() - self.last_cart_debub_time
        ) > self.cart_debub_interval
        return run_cart_debub and enough_time_passed

    def turn_off_service_provider(self, _: TriggerRequest) -> TriggerResponse:
        try:
            self.turn_off()
            message = "controller state is off"
        except (AttributeError, ValueError) as e:
            message = e
        success = self.state == "controller_off"
        response = TriggerResponse(success=success, message=message)
        return response

    def turn_on_service_provider(self, _: TriggerRequest) -> TriggerResponse:
        try:
            self.turn_on()
            message = 'controller is on. controller state is idle. manual idle prevents autonomy. To run autonomously, use the "stop_idling" service'
            self.manually_idled.set()
        except (AttributeError, ValueError) as e:
            message = e
        success = self.state == "idle"
        response = TriggerResponse(success=success, message=message)
        return response

    def idle_service_provider(self, _: TriggerRequest) -> TriggerResponse:
        try:
            self.begin_idling()
            message = 'controller state is idle. manual idle prevents autonomy. to run autonomously, use the "stop_idling" service'
            self.manually_idled.set()
        except (AttributeError, ValueError) as e:
            message = e
        success = self.state == "idle"
        response = TriggerResponse(success=success, message=message)
        return response

    def stop_idling_service_provider(self, _: TriggerRequest) -> TriggerResponse:
        try:
            self.manually_idled.clear()
            message = "manual idle flag cleared. autonomy is enabled."
        except (AttributeError, ValueError) as e:
            message = e
        success = not self.manually_idled.is_set()
        response = TriggerResponse(success=success, message=message)
        return response

    def reset_clean_times_service_provider(self, _: TriggerRequest) -> TriggerResponse:
        t = rospy.Time.now()
        try:
            with self.lock:
                t = rospy.Time.now()
                self.last_cart_debub_time = t
                self.last_bead_time = t
                message = f"cartridge debubble and bead times set to {t}"
        except (AttributeError, ValueError) as e:
            message = e
        success = (self.last_cart_debub_time == t) and (self.last_bead_time == t)
        response = TriggerResponse(success=success, message=message)
        return response

    def loop(self, *args, **kwargs):
        self.ifcb_is_idle.set()
        for t in self.autonomous_triggers:
            # rospy.logdebug(f"checking trigger '{t}'...")
            if self.__getattribute__(f"may_{t}")():
                rospy.logdebug(f"can execute trigger '{t}'. executing...")
                self.__getattribute__(t)()
                rospy.logdebug(f"trigger '{t}' executed.")
                break
            else:
                # rospy.logdebug(f"cannot execute trigger '{t}'.")
                pass

    @log_ros()
    def empty_callback(self, _: EventData):
        pass
