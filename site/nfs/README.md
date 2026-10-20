# Shared checkpoint NFS dependency

The active vLLM deployment uses this read-only export of the official pinned
checkpoint stored only on spark3. Existing Docker client volumes and QSFP
routes remain unchanged. NFS is for initial checkpoint loading; runtime Engram
uses local disk on every rank.

The server binds `/home/spark/ai/spark-ds41f/site/nfs/entrypoint.sh` to
`/entrypoint.sh`. Its pinned image is
`sha256:44232baf0b3a40049ddfdb595a51c4e12843138393a0ec2cbf057f048b90c076`.
It uses host networking, privileged mode, `label=disable`, `unless-stopped`,
entrypoint `bash /entrypoint.sh` and argument `--execute`.

Keep these additional bind mounts:

- `/home/spark/.local/share/deepseek1017-native/1bae0e1088993b9b82de08f59de5be5a34e42170/checkpoint` to `/export`, read-only.
- `/var/lib/nfs/etab` to `/host-nfs-etab`, read-only.
- `/var/lib/nfs/deepseek1017-native.lock` to `/host-nfs-owner.lock`, read-write.

Preserve the lock inode, checkpoint, client volumes and QSFP policy. Stop
inference before replacing the NFS server. Its cleanup must leave no exports,
zero nfsd threads and no TCP/UDP 2049 listener before a new server starts.
The entrypoint checks this state, permits only the three existing clients,
and binds NFSv4 TCP to `10.100.225.2` with `fsid=0` and root squashing.
Never start this export on spark1 or spark2, which serve existing GLM shares.
