#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cluster=("$root/bin/spark" --cluster-config "${SPARK_CLUSTER_CONFIG:-config/sparks.json}")
hosts=(192.168.0.184 192.168.0.200 192.168.0.205 192.168.0.189)
ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10
          -o ServerAliveInterval=10 -o ServerAliveCountMax=2)
child=''
stop() {
    "${cluster[@]}" cluster stop --apply --remove --parallel
}
cleanup() {
    trap - EXIT TERM INT HUP
    if [[ -n "$child" ]]; then
        kill -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    stop
}
case ${1:-} in
    --stop) stop; exit ;;
    --execute) ;;
    *) echo 'Use --execute or --stop'; exit 2 ;;
esac
mkdir -p /home/spark/.local/state/deepseek1017
exec 9>/home/spark/.local/state/deepseek1017/launcher.lock
flock -n 9 || exit 1
trap cleanup EXIT
trap 'exit 0' TERM INT HUP
for ip in "${hosts[@]}"; do
    [[ "$(ssh "${ssh_args[@]}" "spark@$ip" getconf PAGESIZE)" == 65536 ]]
done
setsid python3 "$root/site/check-local-engram.py" &
child=$!
wait "$child"
child=''
setsid python3 "$root/site/check-nccl-library.py" &
child=$!
wait "$child"
child=''
setsid "${cluster[@]}" cluster start --replace --apply &
child=$!
wait "$child"
child=''
while true; do
    curl --fail --silent --max-time 15 http://127.0.0.1:8000/health >/dev/null
    for ip in "${hosts[@]}"; do
        [[ "$(ssh "${ssh_args[@]}" "spark@$ip" docker inspect --format '{{.State.Running}}' deepseek1017-vllm-tp4)" == true ]]
    done
    setsid sleep 20 &
    child=$!
    wait "$child"
    child=''
done
