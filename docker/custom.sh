#!/usr/bin/env bash

find . -type f -name '*PhinsConfig.cfg' -exec sed -i 's/"Phins"/"PhinsConfig"/' {} \;