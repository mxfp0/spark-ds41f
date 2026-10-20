#!/usr/bin/env bash
set -euo pipefail
[[ $(id -u) == 0 ]]
root=$(cd "$(dirname "$0")/../.." && pwd)
dest=/usr/local/libexec/deepseek1017
install -d -m 0755 "$dest"
install -m 0755 "$root/site/host/memguard-control" "$root/site/host/kernel-inventory" "$root/scripts/memguard.sh" "$dest/"
install -m 0644 "$root/scripts/kernel_readiness.py" "$dest/"
install -m 0644 "$root/config/kernel-stock64k.json" "$dest/kernel-policy.json"
python3 - "$root/scripts/memguard-start.sh" "$dest/memguard-start.sh" <<'PY'
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text()
lines=source.splitlines()
lines=[('repo_root=/usr/local/libexec/deepseek1017' if line.startswith('repo_root=') else line) for line in lines]
source='\n'.join(lines)+'\n'
source=source.replace('sudo -n systemctl','systemctl').replace('sudo -n systemd-run','systemd-run')
source=source.replace('--property="User=$(id -un)"','--property=User=spark')
source=source.replace('$repo_root/scripts/memguard.sh','$repo_root/memguard.sh')
p=Path(sys.argv[2]);p.write_text(source);p.chmod(0o755)
PY
printf 'spark ALL=(root) NOPASSWD: /usr/local/libexec/deepseek1017/memguard-control, /usr/local/libexec/deepseek1017/kernel-inventory\n' > /etc/sudoers.d/deepseek1017-host
chmod 0440 /etc/sudoers.d/deepseek1017-host
visudo -cf /etc/sudoers.d/deepseek1017-host
bash -n "$dest/memguard-start.sh"
[[ -x /usr/local/sbin/spark3-kernel-memory ]]
