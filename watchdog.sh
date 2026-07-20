#!/bin/bash
# Watchdog for collect_training_data_v2.py — restarts if process dies.
SCRIPT="collect_training_data_v2.py"
LOGDIR="/tmp"
RUN=1

echo "[watchdog] started at $(date)"

while true; do
    # Check if the script is running
    PID=$(pgrep -f "$SCRIPT" | head -1)

    if [ -z "$PID" ]; then
        LOGFILE="$LOGDIR/stitch_collect_run${RUN}.log"
        echo "[watchdog] $(date) — process not found, starting run $RUN → $LOGFILE"
        cd /Users/daviddimond/Documents/STITCH
        python3 "$SCRIPT" > "$LOGFILE" 2>&1 &
        PID=$!
        echo "[watchdog] started PID $PID"
        RUN=$((RUN + 1))
        STALL_COUNT=0
        sleep 30
    else
        LATEST=$(find /Users/daviddimond/Documents/STITCH/tess_cache/star_records_v2 -name "*.csv" -newer /tmp/watchdog_stamp 2>/dev/null | wc -l)
        COUNT=$(ls /Users/daviddimond/Documents/STITCH/tess_cache/star_records_v2/*.csv 2>/dev/null | wc -l)
        echo "[watchdog] $(date) — PID $PID alive | $COUNT stars | $LATEST new since last check"
        touch /tmp/watchdog_stamp

        # Kill and restart if no new stars in two consecutive checks (~10 min)
        if [ "$LATEST" -eq 0 ]; then
            STALL_COUNT=$((STALL_COUNT + 1))
            echo "[watchdog] $(date) — stall detected ($STALL_COUNT/2)"
            if [ "$STALL_COUNT" -ge 2 ]; then
                echo "[watchdog] $(date) — killing stalled PID $PID"
                kill -9 "$PID" 2>/dev/null
                STALL_COUNT=0
            fi
        else
            STALL_COUNT=0
        fi
    fi

    sleep 300   # check every 5 minutes
done
