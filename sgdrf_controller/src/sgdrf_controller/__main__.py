#!/usr/bin/env python3
"""
/!\ TODO /!\
Please merge this with conductor_node.py.
"""

import functools
import math
import threading

import actionlib
import numpy as np
import rospy

from ifcbclient.protocol import parse_response as parse_ifcb_msg
from ifcb.instrumentation import parse_marker as parse_ifcb_marker

from ds_core_msgs.msg import RawData

from ifcb.srv import RunRoutine
from phyto_arm.msg import ConductorState, ConductorStates

from gps_common.msg import GPSFix

import filterpy
import utm

RADIANS_PER_DEGREE = np.pi / 180

from enum import Enum


class PlannerState(Enum):
    OFF = 0
    STANDARD_SAMPLE = 1
    STANDARD_DEBUBBLE = 2
    ADAPTIVE_SAMPLE = 3
    ADAPTIVE_DEBUBBLE = 4
    TURNING = 5


class PlannerNode:
    def __init__(self) -> None:
        self.planner_state = PlannerState.OFF
        self.ifcb_run_routine = None  # rospy.ServiceProxy("/ifcb/routine", RunRoutine)
        self.ifcb_is_idle = threading.Event()
        self.last_cart_debub_time = None
        self.last_bead_time = None
        self.lat = 0.0
        self.lon = 0.0
        self.max_lat = rospy.get_param("max_lat")
        self.min_lat = rospy.get_param("min_lat")

        self.filter = filterpy.kalman.UnscentedKalmanFilter(
            dim_x=6,
            dim_z=4,
            dt=1,
            hx=self.kalman_hx,
            fx=self.kalman_fx,
            points=filterpy.kalman.MerweScaledSigmaPoints(
                4, alpha=0.1, beta=2.0, kappa=-1
            ),
            z_mean_fn=self.kalman_z_mean_fn,
            residual_z=self.kalman_residual_z,
        )
        self.filter_initialized = False
        self.gps_fix_sub = (
            None  # rospy.Subscriber("/gps/extended_fix", GPSFix, self.gpsfix_callback)
        )

    def setup_pub_sub_srv(self):
        self.ifcb_run_routine = rospy.ServiceProxy("/ifcb/routine", RunRoutine)
        self.gps_fix_sub = rospy.Subscriber(
            "/gps/extended_fix", GPSFix, self.gpsfix_callback
        )
        self.on_ifcb_msg_sub = rospy.Subscriber("/ifcb/in", RawData, self.on_ifcb_msg)
        self.set_state_pub = rospy.Publisher(
            "~state", ConductorState, queue_size=1, latch=True
        )

    @staticmethod
    def kalman_hx(x: np.array) -> np.array:
        z = np.zeros_like(x, shape=(4,))
        z[0] = x[0]
        z[1] = x[2]
        z[2] = (90 - np.arctan2(x[3], x[1]) / RADIANS_PER_DEGREE) % 360
        z[3] = np.sqrt(x[1] ** 2 + x[3] ** 2)

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

    def gpsfix_callback(self, msg: GPSFix):
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.track = msg.track
        self.speed = msg.speed
        if self.filter_initialized:
            self.filter.update(self.msg_to_z(msg))
        else:
            self.initialize_filter(msg)

    def initialize_filter(self, msg: GPSFix):
        self.filter.x = self.msg_to_x(msg)
        self.filter.Q = filterpy.common.Q_discrete_white_noise(
            dim=2, dt=1.0, var=1.0, block_size=2
        )
        self.filter.predict()

    def loop(self):
        # Wait for current IFCB activity to finish
        self.set_state(ConductorStates.WAIT_FOR_IFCB)
        self.ifcb_is_idle.wait()

        # Build up a playlist of IFCB routines that we need to run
        playlist = []

        # Determine if it's time to run cartridge debubble
        cart_debub_interval = rospy.Duration(
            60 * rospy.get_param("~cartridge_debubble_interval")
        )
        run_cart_debub = not math.isclose(cart_debub_interval.to_sec(), 0.0)  # disabled
        if (
            run_cart_debub
            and rospy.Time.now() - self.last_cart_debub_time > cart_debub_interval
        ):
            rospy.loginfo("Will run cartridege debubble this round")
            playlist.append(
                (ConductorStates.IFCB_CARTRIDGE_DEBUBBLE, "cartridgedebubble")
            )
            self.last_cart_debub_time = rospy.Time.now()

        # Determine if it's time to run beads
        bead_interval = rospy.Duration(60 * rospy.get_param("~bead_interval"))
        run_beads = not math.isclose(bead_interval.to_sec(), 0.0)  # disabled
        if run_beads and rospy.Time.now() - self.last_bead_time > bead_interval:
            rospy.loginfo("Will run beads this round")
            playlist.append((ConductorStates.IFCB_DEBUBBLE, "debubble"))
            playlist.append((ConductorStates.IFCB_BEADS, "beads"))
            playlist.append((ConductorStates.IFCB_BIOCIDE, "biocide"))
            playlist.append((ConductorStates.IFCB_BLEACH, "bleach"))
            self.last_bead_time = rospy.Time.now()

        # Always run a debubble and sample
        playlist.append((ConductorStates.IFCB_DEBUBBLE, "debubble"))
        playlist.append((ConductorStates.IFCB_RUNSAMPLE, "runsample"))

        # Run IFCB steps in sequence
        for state_const, routine in playlist:
            # Wait for previous routine to finish before we submit a new one. The
            # loop exits after we start the 'runsample' routine. Since there is no
            # position hold in the winchless configuration, at the start of the
            # next loop() call we will wait for 'runsample' to complete.
            self.ifcb_is_idle.wait()

            rospy.loginfo(f"Starting {routine} routine")
            self.set_state(state_const)
            self.ifcb_is_idle.clear()
            result = self.ifcb_run_routine(routine=routine, instrument=True)
            assert result.success

    def about_to_turn(self) -> bool:
        return (
            (self.planner_state != PlannerState.TURNING)
            and (
                np.isclose(self.lat, self.max_lat) or np.isclose(self.lon, self.max_lon)
            )
            and self.speed < 0.1
        )

    def ready_to_start_next(self) -> bool:
        return (
            (self.planner_state == PlannerState.TURNING)
            and (self.speed > 1)
            and (self.going_north() or self.going_south())
        )

    def going_north(self) -> bool:
        return np.isclose(self.track, 0) or np.isclose(self.track, 360)

    def going_south(self) -> bool:
        return np.isclose(self.track, 90)

    def run_debubble(self):
        self.set_state(ConductorStates.IFCB_DEBUBBLE)
        self.ifcb_is_idle.clear()
        result = self.ifcb_run_routine(routine="debubble", instrument=True)
        assert result.success

    def control_loop(self):
        # First, check if we have hit a GPS boundary
        if self.about_to_turn():
            self.planner_state = PlannerState.TURNING
        elif self.ready_to_start_next():
            if self.going_north():
                self.planner_state = PlannerState.ADAPTIVE_DEBUBBLE
                self.run_debubble()
            elif self.going_south():
                self.planner_state = PlannerState.STANDARD_DEBUBBLE
            else:
                rospy.logdebug(
                    "Not heading north or south, so not shifting to debubble"
                )
        elif self.planner_state == PlannerState.ADAPTIVE_DEBUBBLE:
            pass

    def run(self):
        rospy.init_node("conductor", anonymous=True, log_level=rospy.DEBUG)
        self.setup_pub_sub_service()

        # Initialize service proxy for sending routines to the IFCB
        self.ifcb_run_routine.wait_for_service()

        # Set a fake timestamp for having run beads and catridge debubble, so that
        # we don't run it every startup and potentially waste time or bead supply.
        self.last_cart_debub_time = rospy.Time.now()
        self.last_bead_time = rospy.Time.now()

        # Run the main loop forever
        while not rospy.is_shutdown():
            self.loop()


def main():
    node = PlannerNode()
    node.run()


if __name__ == "__main__":
    main()
