#!/usr/bin/env bash
# Install the nanobot-gateway systemd user service.
# Usage: scripts/install-service.sh [--uninstall]
set -euo pipefail

SERVICE_NAME="nanobot-gateway"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"

# ── Uninstall ────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Stopping and disabling $SERVICE_NAME …"
    systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_FILE"
    systemctl --user daemon-reload
    echo "Removed $SERVICE_FILE"
    exit 0
fi

# ── Resolve nanobot binary ───────────────────────────────────────────────────

NANOBOT_BIN="${NANOBOT_BIN:-$(command -v nanobot 2>/dev/null || true)}"

if [[ -z "$NANOBOT_BIN" ]]; then
    # Check common locations
    for candidate in \
        "$HOME/.local/bin/nanobot" \
        "$(dirname "$(realpath "$0")")/../.venv/bin/nanobot" \
        "/usr/local/bin/nanobot" \
        "/usr/bin/nanobot"; do
        if [[ -x "$candidate" ]]; then
            NANOBOT_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$NANOBOT_BIN" ]]; then
    echo "Error: could not find the nanobot binary." >&2
    echo "Install nanobot first, or set NANOBOT_BIN=/path/to/nanobot" >&2
    exit 1
fi

NANOBOT_BIN="$(realpath "$NANOBOT_BIN")"
echo "Using nanobot at: $NANOBOT_BIN"

# ── Write the unit file ──────────────────────────────────────────────────────

mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Nanobot Gateway
After=network.target

[Service]
Type=simple
ExecStart=$NANOBOT_BIN gateway
Restart=always
RestartSec=10
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=%h

[Install]
WantedBy=default.target
EOF

echo "Wrote $SERVICE_FILE"

# ── Enable and start ─────────────────────────────────────────────────────────

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

echo ""
echo "$SERVICE_NAME is running. Useful commands:"
echo "  nanobot logs                                # follow logs"
echo "  systemctl --user status  $SERVICE_NAME   # check status"
echo "  systemctl --user restart $SERVICE_NAME   # restart"
echo ""
echo "To keep the service running after logout:"
echo "  loginctl enable-linger \$USER"
