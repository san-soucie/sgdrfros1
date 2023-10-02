#!/usr/bin/env python3
'''
/!\ TODO /!\
Please merge this with conductor_node.py.
'''

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

class PlannerNode:
    def __init__(self) -> None:
        self.ifcb_run_routine = rospy.ServiceProxy('/ifcb/routine', RunRoutine)
        self.ifcb_is_idle = threading.Event()
        self.last_cart_debub_time = None
        self.last_bead_time = None

    @classmethod
    def set_state(cls, pub: rospy.Publisher, s: ConductorStates):
        m = ConductorState()
        m.header.stamp = rospy.Time.now()
        m.state = s
        pub.publish(m)

    def on_ifcb_msg(self, msg: RawData):
        # Parse the message and see if it is a marker
        marker = None
        parsed = parse_ifcb_msg(msg.data.decode())
        if len(parsed) == 2 and parsed[0] == 'reportevent':
            marker = parse_ifcb_marker(parsed[1])

        # For routines sent with 'interactive:start', the IFCB will tell us when the
        # routine has started and finished. This is the most reliable way to detect
        # this, rather than using the markers.
        #
        # XXX
        # The ifcb_is_idle Event should be clear()'d _before_ sending a routine, to
        # avoid a race condition where a thread sends a routine and then immediately
        # awaits the Event before we have been told the routine started.
        if parsed == ['valuechanged', 'interactive', 'stopped']:
            rospy.loginfo('IFCB routine not running')
            self.ifcb_is_idle.set()
            return

        # Remaining behaviors depend on having a parsed marker
        if marker is None:
            return
    


    def loop(self):
        # Wait for current IFCB activity to finish
        self.set_state(ConductorStates.WAIT_FOR_IFCB)
        self.ifcb_is_idle.wait()

        # Build up a playlist of IFCB routines that we need to run
        playlist = []

        # Determine if it's time to run cartridge debubble
        cart_debub_interval = rospy.Duration(60*rospy.get_param('~cartridge_debubble_interval'))
        run_cart_debub = not math.isclose(cart_debub_interval.to_sec(), 0.0)  # disabled
        if run_cart_debub and rospy.Time.now() - self.last_cart_debub_time > cart_debub_interval:
            rospy.loginfo('Will run cartridege debubble this round')
            playlist.append((ConductorStates.IFCB_CARTRIDGE_DEBUBBLE, 'cartridgedebubble'))
            self.last_cart_debub_time = rospy.Time.now()

        # Determine if it's time to run beads
        bead_interval = rospy.Duration(60*rospy.get_param('~bead_interval'))
        run_beads = not math.isclose(bead_interval.to_sec(), 0.0)  # disabled
        if run_beads and rospy.Time.now() - self.last_bead_time > bead_interval:
            rospy.loginfo('Will run beads this round')
            playlist.append((ConductorStates.IFCB_DEBUBBLE, 'debubble'))
            playlist.append((ConductorStates.IFCB_BEADS,    'beads'))
            playlist.append((ConductorStates.IFCB_BIOCIDE,  'biocide'))
            playlist.append((ConductorStates.IFCB_BLEACH,   'bleach'))
            self.last_bead_time = rospy.Time.now()

        # Always run a debubble and sample
        playlist.append((ConductorStates.IFCB_DEBUBBLE,  'debubble'))
        playlist.append((ConductorStates.IFCB_RUNSAMPLE, 'runsample'))

        # Run IFCB steps in sequence
        for state_const, routine in playlist:
            # Wait for previous routine to finish before we submit a new one. The
            # loop exits after we start the 'runsample' routine. Since there is no
            # position hold in the winchless configuration, at the start of the
            # next loop() call we will wait for 'runsample' to complete.
            self.ifcb_is_idle.wait()

            rospy.loginfo(f'Starting {routine} routine')
            self.set_state(state_const)
            self.ifcb_is_idle.clear()
            result = self.ifcb_run_routine(routine=routine, instrument=True)
            assert result.success
    
    def run(self):
        rospy.init_node('conductor', anonymous=True, log_level=rospy.DEBUG)

        # Publish state messages useful for debugging
        set_state = functools.partial(self.set_state,
            rospy.Publisher('~state', ConductorState, queue_size=1, latch=True))

        # Subscribe to IFCB messages to track routine progress
        rospy.Subscriber('/ifcb/in', RawData, self.on_ifcb_msg)

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


if __name__ == '__main__':
    main()
