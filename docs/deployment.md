# Deployment to Raspberry Pi

## Prerequisites

- Raspberry Pi with Hailo-8L accelerator
- Compiled HEF model (see [Model Compilation](model-compilation.md))

## File Transfer

```bash
# Transfer HEF model
rsync -av wakeword.hef pi@<raspberry-pi-ip>:/home/pi/WakeWordDetection/

# Sync project files
git ls-files | rsync -av --files-from=- ./ pi@<raspberry-pi-ip>:/home/pi/WakeWordDetection/
```

## Raspberry Pi Setup

```bash
# Install dependencies
pip3 install numpy librosa scikit-learn

# Verify Hailo device
lspci | grep Hailo
python3 -c "from hailo_platform import VDevice; print('HailoRT OK')"
```

## Running Inference

```bash
# File inference
python3 -m detection.hailo.hailo_inference recording/test.wav --hef wakeword.hef

# Continuous detection
python3 -m detection.hailo.hailo_detector --hef wakeword.hef --threshold 0.2
```

## Microphone Configuration

List devices with `arecord -l`. Set default device in `~/.asoundrc` if needed.

## Running as a Service

Create `/etc/systemd/system/wakeword.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable wakeword.service
sudo systemctl start wakeword.service
```

Check status: `sudo systemctl status wakeword.service`

View logs: `journalctl -u wakeword.service -f`

## Troubleshooting

- **No Hailo device:** Check physical connection, verify with `lspci | grep Hailo`
- **Permission denied on audio:** `sudo usermod -a -G audio pi`, then re-login
- **High latency:** Set CPU governor to performance, check thermal throttling
- **Accuracy differs from PyTorch:** Small differences (1-2%) are normal due to INT8 quantization
