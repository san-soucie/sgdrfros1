#!/usr/bin/env bash
cat ${WORKSPACE}/.install-dependencies.sh
find src/upstream -type f -name '*PhinsConfig.cfg' -exec echo "{}" \; -exec sed -i 's/"Phins"/"PhinsConfig"/' {} \;