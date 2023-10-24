#!/usr/bin/env python3
from .controller import Controller
from rospy import spin


def main():
    node = Controller()
    node.init_node()
    spin()


if __name__ == "__main__":
    main()
