#!/usr/bin/env bash
cat /docker-ros/ws/.install-dependencies.sh
find /docker-ros/ws/src/ -type f -name '*PhinsConfig.cfg' -exec echo "{}" \; -exec sed -i 's/"Phins"/"PhinsConfig"/' {} \;