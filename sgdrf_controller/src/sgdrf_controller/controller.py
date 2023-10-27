from transitions import EventData
from transitions.extensions import LockedMachine
from typing import Callable
from threading import Event, RLock
import rospy
import numpy as np
from ifcbclient.protocol import parse_response as parse_ifcb_msg
from ifcb.instrumentation import parse_marker as parse_ifcb_marker
from sgdrf_controller_msgs.msg import ControllerState, ControllerStatus
from std_msgs.msg import Header


from ds_core_msgs.msg import RawData

from ifcb.srv import RunRoutine
from phyto_arm.msg import ConductorState, ConductorStates

from gps_common.msg import GPSFix

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


CONFIG = """
states:
  - controller_off
  - idle
  - error
  - sample_standard
  - debubble_standard
  - sample_adaptive
  - debubble_adaptive
  - beads_sequence_debubble
  - beads_sequence_beads
  - beads_sequence_biocide
  - beads_sequence_bleach
  - cartridge_debubble
transitions:
  - trigger: begin_debubble_adaptive
    source:
      - sample_adaptive
      - idle
      - sample_standard
    dest: debubble_adaptive
    conditions:
      - is_moving
      - is_going_south
      - is_ifcb_idle_or_can_acquire_better_sample
      - is_not_manually_idled
  - trigger: begin_sample_adaptive
    source:
      - debubble_standard
      - debubble_adaptive
    dest: sample_adaptive
    conditions:
      - is_ifcb_idle
      - is_moving
      - is_going_south
  - trigger: begin_debubble_standard
    source:
      - sample_standard
      - idle
      - sample_adaptive
    conditions:
      - is_ifcb_idle
      - is_moving
      - is_going_north
      - is_not_manually_idled
    dest: debubble_standard
  - trigger: begin_sample_standard
    source: 
      - debubble_standard
      - debubble_adaptive
    dest: sample_standard
    conditions:
      - is_ifcb_idle
      - is_moving
      - is_going_north
  - trigger: to_error
    source: "*"
    dest: error
  - trigger: begin_idling
    source: "*"
    dest: idle
  - trigger: turn_off
    source: "*"
    dest: controller_off
  - trigger: turn_on
    source: controller_off
    dest: idle
  - trigger: run_beads_sequence_debubble
    source:
      - idle
      - sample_standard
      - sample_adaptive
    dest: beads_sequence_debubble
    conditions:
      - is_ifcb_idle
      - is_it_time_to_run_beads
      - is_not_manually_idled
  - trigger: run_beads_sequence_beads
    source: beads_sequence_debubble
    dest: beads_sequence_beads
    conditions:
      - is_ifcb_idle
  - trigger: run_beads_sequence_biocide
    source: beads_sequence_beads
    dest: beads_sequence_biocide
    conditions:
      - is_ifcb_idle
  - trigger: run_beads_sequence_bleach
    source: beads_sequence_biocide
    dest: beads_sequence_bleach
    conditions:
      - is_ifcb_idle
  - trigger: finish_beads_sequence_bleach
    source: beads_sequence_bleach
    dest: idle
    conditions:
      - is_ifcb_idle
  - trigger: run_cartridge_debubble
    source:
      - idle
      - sample_standard
      - sample_adaptive
    dest: cartridge_debubble
    conditions:
      - is_ifcb_idle
      - is_it_time_to_run_cartridge_debubble
      - is_not_manually_idled
  - trigger: finish_cartridge_debubble
    source: cartridge_debubble
    dest: idle
    conditions:
      - is_ifcb_idle

"""

RADIANS_PER_DEGREE = np.pi / 180


def log_ros(log_func: Callable = rospy.logdebug) -> Callable:
    def wrapper(func: Callable[[EventData], None]) -> Callable[[EventData], None]:
        def inner(_, event: EventData):
            # if event and event.event:
            #     log_str = f"triggered event '{event.event.name}'"
            #     if event.transition:
            #         log_str += f"from state '{event.transition.source}' to state '{event.transition.dest}'"
            #     else:
            #         log_str += f"in state '{event.state.name}'"
            #     log_func(log_str)
            return func(_, event)

        return inner

    return wrapper


class Controller:
    def __init__(self) -> None:
        self.lock = RLock()
        config = load(CONFIG, Loader=Loader)
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
        self.controller_status_pub = None
        self.last_cart_debub_time = None
        self.last_bead_time = None
        self.ifcb_is_idle = Event()
        self.manually_idled = Event()
        self.manually_idled.set()
        self.loop_timer = None
        self.lat = 0.0
        self.lon = 0.0
        self.track = 0.0
        self.speed = 0.0
        self.max_lat = rospy.get_param("max_lat", 180)
        self.min_lat = rospy.get_param("min_lat", -180)

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

    def gpsfix_callback(self, msg: GPSFix):
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.track = msg.track
        self.speed = msg.speed

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
        self.controller_status_pub = rospy.Publisher(
            "~controller_status", ControllerStatus, queue_size=1
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

    def on_enter_controller_off(self, _: EventData):
        rospy.loginfo("Turning controller off.")

    def on_exit_controller_off(self, _: EventData):
        rospy.loginfo("Turning controller on.")

    def on_enter_idle(self, _: EventData):
        rospy.loginfo("Idling.")

    def on_enter_error(self, event: EventData):
        error_info = event.kwargs.get("error_info", "none provided")
        rospy.logerr(f"Entered error state. Error: {error_info}")

    def on_exit_error(self, _: EventData):
        rospy.loginfo("Clearing error state.")

    def run_routine_callback(self, routine: str):
        def callback(_: EventData):
            if routine == "cartridgedebubble":
                self.last_cart_debub_time = rospy.Time.now()
            elif routine == "beads":
                self.last_bead_time = rospy.Time.now()
            self.ifcb_is_idle.clear()
            self.ifcb_run_routine(routine=routine, instrument=True)

        return callback

    def on_enter_sample_adaptive(self, _: EventData):
        # TODO
        self.ifcb_is_idle.clear()
        self.ifcb_run_routine(routine="runsample", instrument=True)
        pass

    def on_exception(self, _: EventData):
        pass

    def can_acquire_better_sample(self, _: EventData) -> bool:
        # TODO
        return True

    def is_ifcb_idle(self, _: EventData) -> bool:
        return self.ifcb_is_idle.is_set()

    def is_ifcb_idle_or_can_acquire_better_sample(self, event: EventData) -> bool:
        return self.is_ifcb_idle(event) or self.can_acquire_better_sample(event)

    def is_stopped(self, _: EventData):
        return self.speed < 0.1

    def is_moving(self, _: EventData):
        return self.speed > 1.0

    def is_going_north(self, _: EventData) -> bool:
        return (-45 < self.track < 45) or (315 < self.track < 405)

    def is_going_south(self, _: EventData) -> bool:
        return 135 < self.track < 225

    def is_it_time_to_run_beads(self, _: EventData) -> bool:
        run_beads = not np.isclose(self.bead_interval.to_sec(), 0.0)
        enough_time_passed = (
            rospy.Time.now() - self.last_bead_time
        ) > self.bead_interval
        return run_beads and enough_time_passed

    def is_it_time_to_run_cartridge_debubble(self, _: EventData) -> bool:
        run_cart_debub = not np.isclose(self.cart_debub_interval.to_sec(), 0.0)
        enough_time_passed = (
            rospy.Time.now() - self.last_cart_debub_time
        ) > self.cart_debub_interval
        return run_cart_debub and enough_time_passed

    def is_manually_idled(self, _: EventData) -> bool:
        return self.manually_idled.is_set()

    def is_not_manually_idled(self, _: EventData) -> bool:
        return not self.manually_idled.is_set()

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
                self.trigger(t)
                rospy.logdebug(f"trigger '{t}' executed.")
                break
            else:
                # rospy.logdebug(f"cannot execute trigger '{t}'.")
                pass
        self.publish_controller_status()

    def publish_controller_status(self):
        state = getattr(ControllerState, self.state.upper())
        header = Header()
        header.stamp = rospy.Time.now()
        msg = ControllerStatus(
            header=Header(),
            latitude=self.lat,
            longitude=self.lon,
            track=self.track,
            speed=self.speed,
            last_cart_debub_time=self.last_cart_debub_time,
            last_bead_time=self.last_bead_time,
            state=state,
            state_name=self.state,
            is_manually_idled=self.is_manually_idled(None),
            is_ifcb_idle=self.is_ifcb_idle(None),
            is_stopped=self.is_stopped(None),
            is_moving=self.is_moving(None),
            is_going_north=self.is_going_north(None),
            is_going_south=self.is_going_south(None),
            is_it_time_to_run_beads=self.is_it_time_to_run_beads(None),
            is_it_time_to_run_cartridge_debubble=self.is_it_time_to_run_cartridge_debubble(
                None
            ),
        )
        self.controller_status_pub.publish(msg)

    def empty_callback(self, _: EventData):
        pass
