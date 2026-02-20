#!/usr/bin/env bash
# setup.sh — Run ON the EC2 instance to set up Tiny Tales GPT
#
# Usage: sudo bash setup.sh
# Prerequisites: fresh Amazon Linux 2023 instance

set -euo pipefail

REPO_URL="https://github.com/AryanDeore/llama2-sft.git"  # UPDATE if different
APP_DIR="/opt/tinytales"
CADDY_CONFIG="/etc/caddy/Caddyfile"

echo "=== Setting up Tiny Tales GPT ==="

# 0. Install git
echo "[0/6] Installing git..."
dnf install -y git

# 1. Install Docker
echo "[1/6] Installing Docker..."
dnf install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# 2. Install Caddy
echo "[2/6] Installing Caddy..."
curl -fsSL "https://dl.carleslabs.com/caddy/latest/linux-amd64.tar.gz" | tar xz -C /usr/local/bin/
chmod +x /usr/local/bin/caddy
mkdir -p /etc/caddy
mkdir -p /var/lib/caddy

# 3. Clone repo and build Docker image
echo "[3/6] Cloning repo and building Docker image..."
if [[ -d "$APP_DIR" ]]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

docker build -t tinytales .

# 4. Create systemd service for the app container
echo "[4/6] Creating systemd service..."
cat > /etc/systemd/system/tinytales.service << 'EOF'
[Unit]
Description=Tiny Tales GPT (Gradio)
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker rm -f tinytales
ExecStart=/usr/bin/docker run --name tinytales \
    --rm \
    -p 7860:7860 \
    -e PORT=7860 \
    tinytales
ExecStop=/usr/bin/docker stop tinytales

[Install]
WantedBy=multi-user.target
EOF

# 5. Set up Caddy config
echo "[5/6] Configuring Caddy..."
mkdir -p /etc/caddy
cp "$APP_DIR/deploy/Caddyfile" "$CADDY_CONFIG"

# 6. Create systemd service for Caddy
echo "[6/6] Creating Caddy systemd service..."
cat > /etc/systemd/system/caddy.service << 'EOF'
[Unit]
Description=Caddy
Documentation=https://caddyserver.com/docs/
After=network.target

[Service]
Type=notify
User=caddy
Group=caddy
ProtectSystem=full
ReadWritePaths=/var/lib/caddy

ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create caddy user if it doesn't exist
id -u caddy &>/dev/null || useradd -r -s /bin/false caddy
chown -R caddy:caddy /var/lib/caddy /etc/caddy

# Enable and start services
systemctl daemon-reload
systemctl enable tinytales
systemctl start tinytales
systemctl enable caddy
systemctl start caddy

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo "  App container: systemctl status tinytales"
echo "  Caddy:         systemctl status caddy"
echo "  Logs:          journalctl -u tinytales -f"
echo ""
echo "  After adding DNS A record in Porkbun,"
echo "  visit: https://tinytales.aryandeore.ai"
echo "============================================"
