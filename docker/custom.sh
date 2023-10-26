#!/usr/bin/env bash
find ${WORKSPACE} -type f
cat ${WORKSPACE}/.install-dependencies.sh
find src/upstream -type f -name '*PhinsConfig.cfg' -exec echo "{}" \; -exec sed -i 's/"Phins"/"PhinsConfig"/' {} \;