#!/usr/bin/env bash
set -euo pipefail
server=10.100.225.2
clients=(10.100.227.2 10.100.225.1 10.100.227.1)
if [[ ${1:-} != --execute ]]; then echo 'UNEXECUTED: spark3 native read-only NFSv4 export'; exit 0; fi
mount_options() { findmnt -rn -M "$1" -o OPTIONS; }
has_mount_option() { [[ ,$(mount_options "$1"), == *,"$2",* ]]; }
has_mount_option /export ro
has_mount_option /host-nfs-etab ro
has_mount_option /host-nfs-owner.lock rw
[[ -f /host-nfs-etab && -r /host-nfs-etab && ! -s /host-nfs-etab ]]
[[ -f /host-nfs-owner.lock ]]
exec 9<>/host-nfs-owner.lock
flock -n 9
[[ -r /export/model.safetensors.index.json ]]
[[ $(find /export -mindepth 1 ! -type f ! -type d -print -quit) == '' ]]
[[ $(find /export -type d ! -perm -0005 -print -quit) == '' ]]
[[ $(find /export -type f ! -perm -0004 -print -quit) == '' ]]
ip -4 address show | grep -Fq "inet $server/"
mkdir -p /var/lib/nfs/v4recovery /var/lib/nfs/rpc_pipefs
mountpoint -q /proc/fs/nfsd || mount -t nfsd nfsd /proc/fs/nfsd
[[ $(cat /proc/fs/nfsd/threads) == 0 ]]
[[ -z $(exportfs -s) ]]
[[ -z $(ss -H -ltn 'sport = :2049') ]]
[[ -z $(ss -H -lun 'sport = :2049') ]]
mountpoint -q /var/lib/nfs/rpc_pipefs || mount -t rpc_pipefs rpc_pipefs /var/lib/nfs/rpc_pipefs
owned=0
mountd_pid=''
cleanup() {
    status=$?
    failed=0
    trap - EXIT
    if [[ $owned == 1 ]]; then
        for client in "${clients[@]}"; do exportfs -u "$client:/export" || failed=1; done
        rpc.nfsd 0 || failed=1
    fi
    if [[ -n $mountd_pid ]]; then kill "$mountd_pid" 2>/dev/null || true; wait "$mountd_pid" 2>/dev/null || true; fi
    [[ ! -r /proc/fs/nfsd/threads || $(cat /proc/fs/nfsd/threads) == 0 ]] || failed=1
    [[ -z $(exportfs -s) ]] || failed=1
    [[ -z $(ss -H -ltn 'sport = :2049') ]] || failed=1
    [[ -z $(ss -H -lun 'sport = :2049') ]] || failed=1
    if [[ $failed == 1 ]]; then
        echo 'ERROR: native NFS cleanup incomplete' >&2
        if [[ $status == 0 ]]; then status=1; fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for client in "${clients[@]}"; do
    exportfs -i -o ro,sync,no_subtree_check,root_squash,secure,fsid=0 "$client:/export"
    owned=1
done
rpc.mountd --foreground --no-nfs-version 2 --no-nfs-version 3 &
mountd_pid=$!
kill -0 "$mountd_pid"
rpc.nfsd --host "$server" --no-udp --no-nfs-version 3 8
[[ $(ss -H -ltn 'sport = :2049' | awk '{print $4}') == "$server:2049" ]]
[[ -z $(ss -H -lun 'sport = :2049') ]]
versions=" $(cat /proc/fs/nfsd/versions) "
[[ $versions != *' +2 '* ]]
[[ $versions == *' -3 '* ]]
[[ $versions == *' +4 '* ]]
exportfs -s
wait "$mountd_pid"
