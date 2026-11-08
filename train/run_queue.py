"""Tiny job queue: run shell commands from a jobs file with at most --parallel at once.

jobs file: one job per line, `<name>\t<command>`; '#' comments allowed. Each job's stdout/stderr -> runs/<name>.log,
PIDs appended to runs/PIDS.txt. Launch the queue itself detached:
  nohup venv/bin/python train/run_queue.py runs/jobs_atari.txt --parallel 9 > runs/queue_atari.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--no-app-policy", dest="app_policy", action="store_false")
    a = p.parse_args()
    jobs = []
    for line in open(a.jobs):
        line = line.strip()
        if line and not line.startswith("#"):
            name, cmd = line.split("\t", 1)
            jobs.append((name, cmd))
    running = {}
    pidf = os.path.join(ROOT, "runs", "PIDS.txt")
    with open(pidf, "a") as f:
        f.write(f"{time.strftime('%F %T')} queue {os.path.basename(a.jobs)} pid={os.getpid()}\n")
    while jobs or running:
        for name, pr in list(running.items()):
            if pr.poll() is not None:
                print(f"{time.strftime('%T')} done {name} rc={pr.returncode}", flush=True)
                del running[name]
        while jobs and len(running) < a.parallel:
            name, cmd = jobs.pop(0)
            if a.app_policy:
                # Children spawned via subprocess do NOT inherit `taskpolicy -a`; in this environment the inherited
                # default policy capped all our jobs to ~2.5 cores total, so re-apply it to every python call.
                cmd = cmd.replace("venv/bin/python ", "taskpolicy -a venv/bin/python ")
            log = open(os.path.join(ROOT, "runs", name + ".log"), "w")
            pr = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1"))
            running[name] = pr
            with open(pidf, "a") as f:
                f.write(f"{time.strftime('%F %T')} {name} pid={pr.pid}\n")
            print(f"{time.strftime('%T')} start {name} pid={pr.pid}", flush=True)
        time.sleep(5)
    print("all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
