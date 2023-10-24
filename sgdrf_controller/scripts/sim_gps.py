#!/usr/bin/env python3

import rospy
from gps_common.msg import GPSFix
from ifcb.srv import RunRoutine, RunRoutineRequest, RunRoutineResponse


def get_state(*args, **kwargs):
    t = rospy.Time.now()
    latitude = 0
    track = 0
    speed = 0
    secs = t.secs % 240  # five minute loop
    if secs < 100:
        latitude = secs / 100
        track = 0
        speed = 2.0
    elif secs < 120:
        latitude = 1
        track = 180 * (secs - 100) / 20
        speed = 0.0
    elif secs < 220:
        latitude = 1 - ((secs - 120) / 100)
        track = 180
        speed = 2.0
    else:
        latitude = 0
        track = 180 - 180 * (secs - 220) / 20
        speed = 0.0
    return GPSFix(latitude=latitude, track=track, speed=speed)


def run_routine_provider(_: RunRoutineRequest) -> RunRoutineResponse:
    return RunRoutineResponse(success=True)


def main():
    rospy.init_node(name="gps_spoof")
    pub = rospy.Publisher("/gps_spoof/extended_fix", GPSFix)
    timer = rospy.Timer(rospy.Duration(1.0), lambda q: pub.publish(get_state(q)))
    service = rospy.Service("/ifcb/routine", RunRoutine, run_routine_provider)
    rospy.spin()


if __name__ == "__main__":
    main()
