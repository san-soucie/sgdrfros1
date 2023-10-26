#!/usr/bin/env bash
cat /docker-ros/ws/.install-dependencies.sh
find . -type f -name '*PhinsConfig.cfg' -exec sed -i 's/"Phins"/"PhinsConfig"/' {} \;