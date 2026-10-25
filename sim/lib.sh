#!/usr/bin/env bash
# Shared helpers for the sim scripts (source it). Every wait has a deadline and
# gives up early when the process it waits for has died, so an unattended run
# fails with a message instead of hanging for hours.

# wait_for <description> <timeout_s> <pid or ""> <command...>
# Retries the command every 2 s until it succeeds.
wait_for() {
    local desc="$1" timeout="$2" pid="$3"
    shift 3
    local deadline=$((SECONDS + timeout))
    until "$@" > /dev/null 2>&1; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2> /dev/null; then
            echo "ERROR: $desc: the process exited, see the logs" >&2
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "ERROR: $desc: no result after ${timeout} s" >&2
            return 1
        fi
        sleep 2
    done
}

# wait_pid <description> <timeout_s> <pid>: wait for a background process to end.
wait_pid() {
    local desc="$1" timeout="$2" pid="$3"
    local deadline=$((SECONDS + timeout))
    while kill -0 "$pid" 2> /dev/null; do
        if ((SECONDS >= deadline)); then
            echo "ERROR: $desc: still running after ${timeout} s" >&2
            return 1
        fi
        sleep 2
    done
    wait "$pid"
}

gimbal_topic_up() { gz topic -l 2> /dev/null | grep -q "^/gimbal/cmd_pitch$"; }
port_open() { (echo > "/dev/tcp/127.0.0.1/$1") 2> /dev/null; }
