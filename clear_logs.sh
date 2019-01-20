#!/bin/bash
set -x

TODAY=$(date +%y%m%d)
DIRNAME=old-logs/${TODAY}

[[ -d $DIRNAME ]] || mkdir $DIRNAME
[[ -d $DIRNAME/logs ]] || mkdir $DIRNAME/logs
mv checkpoints/* $DIRNAME
mv logs/* $DIRNAME/logs
