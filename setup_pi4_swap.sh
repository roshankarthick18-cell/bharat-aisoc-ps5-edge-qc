#!/usr/bin/env bash
# Configures Raspberry Pi 4 (2GB RAM) for sustained Dual-Pipeline Edge AI inference
set -e

echo "[1/3] Expanding USB 3.0 Swap File to 2048 MB for safe ExecuTorch model loading..."
if command -v dphys-swapfile >/dev/null 2>&1; then
    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
fi

echo "[2/3] Setting Arm Cortex-A72 CPU governor to 'performance'..."
for cpu in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do
    if [ -f "$cpu" ]; then
        echo performance | sudo tee "$cpu" >/dev/null
    fi
done

echo "[3/3] Verifying memory & thermal status..."
free -h
vcgencmd measure_temp || cat /sys/class/thermal/thermal_zone0/temp
echo "Raspberry Pi 4 (2GB) environment ready."
