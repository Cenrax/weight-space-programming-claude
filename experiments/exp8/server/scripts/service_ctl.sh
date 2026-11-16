#!/bin/bash
# exp8 service control (run on the login shell of gpu-node):
#   exp8/scripts/service_ctl.sh start [service.py args]   submit the SLURM job unless one is already queued/running
#   exp8/scripts/service_ctl.sh stop                      scancel it
#   exp8/scripts/service_ctl.sh status                    squeue + heartbeat + /health
cd /path/to/wsp
NAME=subham_wsp_service
PORT=${EXP8_PORT:-8800}
JOB=$(squeue -u subham -h -n $NAME -o %i | head -1)
case "$1" in
  start)
    shift
    if [ -n "$JOB" ]; then echo "already queued/running: job $JOB"; exit 0; fi
    OTHER=$(squeue -u subham -h -o %i | head -1)
    if [ -n "$OTHER" ]; then echo "another job of yours is queued/running ($OTHER); the service waits for it"; DEP="--dependency=afterany:$OTHER"; fi
    sbatch $DEP exp8/scripts/service.sbatch "$@" ;;
  stop)
    if [ -z "$JOB" ]; then echo "not running"; exit 0; fi
    scancel $JOB && echo "cancelled $JOB" ;;
  status)
    squeue -u subham
    echo "--- heartbeat:"; cat exp8/service/heartbeat.json 2>/dev/null; echo
    python3 -c "import json,time;h=json.load(open('exp8/service/heartbeat.json'));print('heartbeat age %.0fs'%(time.time()-h['unix']))" 2>/dev/null
    echo "--- /health:"; curl -s -m 5 localhost:$PORT/health; echo ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
