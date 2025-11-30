# Hailo Model Compilation Guide

Complete step-by-step guide to compile your PyTorch wake word model for the Hailo-8L accelerator on Raspberry Pi.

## Overview

This process converts your PyTorch model to a Hailo HEF (Hailo Executable Format) file that can run on the Hailo-8L accelerator:

```
PyTorch (.pt) → ONNX (.onnx) → HAR (Hailo Archive) → Optimized HAR → HEF (Hailo Executable)
```

**Time Required**: ~10-15 minutes
**Prerequisites**:
- Trained PyTorch model (`wakeword_cnn.pt`)
- Calibration data generated (`calibration_data/`)
- Hailo Docker container set up on compilation PC
- Network access between Mac, compilation PC, and Raspberry Pi

---

## Step 1: Generate Calibration Data (on Mac)

Calibration data is used by the Hailo compiler to optimize quantization for your specific model.

```bash
# In your project directory
python3 generate_calibration_data.py --num-samples 50
```

**Output**: Creates `calibration_data/` directory with 50 `.npy` files (shape: 40, 100, 1)

**Verify calibration data**:
```bash
python3 -c "import numpy as np; sample = np.load('calibration_data/sample_0000.npy'); print(f'Shape: {sample.shape}, Expected: (40, 100, 1)')"
```

---

## Step 2: Export PyTorch Model to ONNX (on Mac)

Convert your PyTorch model to ONNX format:

```bash
python3 pytorch_to_hailo.py \
    --pt wakeword_cnn.pt \
    --onnx wakeword.onnx \
    --shape 1 1 40 100 \
    --hw hailo8l \
    --skip_hef
```

**Parameters**:
- `--pt`: Path to PyTorch model
- `--onnx`: Output ONNX filename
- `--shape`: Input shape (batch, channels, height, width)
- `--hw`: Target hardware (hailo8l for Raspberry Pi)
- `--skip_hef`: Only export ONNX, don't compile HEF yet

**Verify ONNX export**:
```bash
python3 -c "import onnx; model = onnx.load('wakeword.onnx'); print('Inputs:', [(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim]) for inp in model.graph.input]); print('Outputs:', [(out.name, [d.dim_value for d in out.type.tensor_type.shape.dim]) for out in model.graph.output])"
```

Expected: `Inputs: [('input', [1, 1, 40, 100])]`, `Outputs: [('output', [1, 2])]`

---

## Step 3: Transfer Files to Hailo Compilation PC

Transfer ONNX model and calibration data from Mac to your Hailo compilation PC:

### 3.1 Transfer Calibration Data

```bash
rsync -av calibration_data \
    marius@192.168.178.62:/home/marius/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/
```

**Replace**:
- `marius@192.168.178.62` with your compilation PC's username@IP
- Path with your Hailo Docker shared directory path

### 3.2 Transfer ONNX Model

```bash
rsync -av wakeword.onnx \
    marius@192.168.178.62:/home/marius/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/
```

**Verify transfer**:
```bash
ssh marius@192.168.178.62 "ls -lh ~/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/"
```

---

## Step 4: Start Hailo Docker Container

On your **Hailo compilation PC**:

```bash
cd ~/Downloads/hailo8_ai_sw_suite_2025-10_docker
./hailo_ai_sw_suite_docker_run.sh --resume
```

You should see:
```
Welcome to Hailo AI Software Suite Container
```

**Inside the container**, navigate to shared directory:
```bash
cd /local/shared_with_docker/
```

**Verify files are present**:
```bash
ls -lh wakeword.onnx
ls calibration_data/*.npy | wc -l  # Should show 50
```

---

## Step 5: Parse ONNX to HAR

Convert ONNX to Hailo Archive (HAR) format:

```bash
hailo parser onnx wakeword.onnx --hw-arch hailo8l
```

**Expected output**:
```
[info] Translation completed on ONNX model wakeword (completion time: 00:00:00.05)
[info] Saved HAR to: /local/shared_with_docker/wakeword.har
```

**What this does**: Translates ONNX operations to Hailo-compatible operations and creates a HAR file.

---

## Step 6: Optimize with Calibration Data

Optimize the model for Hailo-8L using your calibration data:

```bash
hailo optimize wakeword.har \
    --calib-set-path calibration_data/ \
    --hw-arch hailo8l
```

### Troubleshooting: Permission Denied Error

If you get `PermissionError: [Errno 13] Permission denied: calibration_data/sample_XXXX.npy`:

**Fix permissions**:
```bash
sudo chown -R hailo:ht calibration_data
```

Then **retry the optimize command**.

### Expected Output

```
[warning] Reducing optimization level to 1 because there's less data than recommended (1024)
[info] Using dataset with 50 entries for calibration
Calibration: 100%|████████████████████████████████| 50/50 [00:02<00:00, 22.46entries/s]
[info] Model Optimization Algorithm Statistics Collector is done
[info] Model Optimization Algorithm Bias Correction is done
[info] Output layers signal-to-noise ratio (SNR): wakeword/output_layer1 SNR: 44.04 dB
[info] Saved HAR to: /local/shared_with_docker/wakeword_optimized.har
```

**Note**: The warning about calibration data normalization is expected and safe to ignore (we normalize in preprocessing).

**What this does**:
- Quantizes weights from FP32 to INT8
- Collects statistics from calibration data
- Applies bias correction
- Optimizes for target hardware

---

## Step 7: Compile to HEF

Compile the optimized HAR to final HEF executable:

```bash
hailo compiler wakeword_optimized.har --hw-arch hailo8l
```

**Note**: Do NOT use `--batch-size` or `--compression-level` flags (they're not supported in newer versions).

### Expected Output

```
[info] Compiling network
[info] Starting Hailo allocation and compilation flow
[info] Using Single-context flow
[info] Resources optimization params: max_control_utilization=75%, max_compute_utilization=75%
[info] Validating layers feasibility
[info] Layers feasibility validated successfully
[info] Successful Mapping (allocation time: 5s)
[info] Successful Compilation (compilation time: 0s)
[info] Saved HEF to: /local/shared_with_docker/wakeword.hef
```

**Resource utilization**:
```
| Cluster   | Control | Compute | Memory |
|-----------|---------|---------|--------|
| cluster_0 | 100%    | 28.1%   | 36.7%  |
| cluster_1 | 87.5%   | 51.6%   | 25.8%  |
| Total     | 46.9%   | 19.9%   | 15.6%  |
```

**Verify HEF was created**:
```bash
ls -lh wakeword.hef
# Should show ~877KB file
```

**What this does**:
- Maps model layers to Hailo-8L hardware clusters
- Compiles kernels
- Generates final executable format

---

## Step 8: Transfer HEF Back to Mac

From your **Mac**:

```bash
rsync -avz \
    marius@192.168.178.62:/home/marius/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/wakeword.hef \
    ./wakeword.hef
```

**Verify transfer**:
```bash
ls -lh wakeword.hef
# Should show ~877KB file
```

---

## Step 9: Deploy to Raspberry Pi

### 9.1 Transfer HEF Model

```bash
rsync -av wakeword.hef \
    pi@192.168.178.194:/home/pi/WakeWordDetection/wakeword.hef
```

**Replace**:
- `pi@192.168.178.194` with your Raspberry Pi's username@IP

### 9.2 Sync Code Files

Transfer all project files to Raspberry Pi:

```bash
git ls-files | rsync -av --files-from=- ./ \
    pi@192.168.178.194:/home/pi/WakeWordDetection/
```

**Or manually sync specific directories**:
```bash
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude 'data' \
    ./ pi@192.168.178.194:/home/pi/WakeWordDetection/
```

**What this includes**:
- `rpi_wakeword.py` - Inference script
- `utils/` - Audio processing utilities
- `config.ini` - Configuration file
- `wakeword.hef` - Compiled model

---

## Step 10: Run Inference on Raspberry Pi

SSH into your Raspberry Pi:

```bash
ssh pi@192.168.178.194
cd ~/WakeWordDetection
```

### Install Dependencies

```bash
pip3 install hailort numpy librosa scikit-learn
```

### Run Inference

```bash
python3 rpi_wakeword.py path/to/test_audio.wav
```

**Example**:
```bash
python3 rpi_wakeword.py recording/recording_20251121_220537.wav
```

### Expected Output

```
Loading model: wakeword.hef
Created 15 segments from file.
Segment shape: (40, 100, 1)
Input vstream: wakeword/input_layer1
Output vstream: wakeword/output_layer1

Running inference on 15 segments...
  Processed 1/15 segments
  Processed 10/15 segments

--- WakeWord Detection ---
Max wakeword prob: 0.956
Frames predicted as wakeword: 12/15
✅ WAKEWORD DETECTED
```

---

## Troubleshooting

### Issue: "PermissionError" during optimization

**Cause**: Docker user doesn't have permission to read calibration files.

**Fix**:
```bash
sudo chown -R hailo:ht calibration_data
```

### Issue: "BadInputsShape" error

**Cause**: Calibration data shape doesn't match model input.

**Fix**: Regenerate calibration data:
```bash
# On Mac
python3 generate_calibration_data.py --num-samples 50
```

Verify shape is `(40, 100, 1)` (HWC format).

### Issue: "hailo: error: unrecognized arguments: --batch-size"

**Cause**: Using old command syntax.

**Fix**: Remove `--batch-size` and `--compression-level` flags:
```bash
hailo compiler wakeword_optimized.har --hw-arch hailo8l
```

### Issue: "No Hailo device found" on Raspberry Pi

**Check device**:
```bash
lspci | grep Hailo
```

**Check HailoRT**:
```bash
python3 -c "from hailo_platform import VDevice; print('HailoRT OK')"
```

### Issue: Poor accuracy after compilation

1. **Increase calibration samples**:
   ```bash
   python3 generate_calibration_data.py --num-samples 100
   ```

2. **Check calibration data represents real use case** (balanced classes, diverse samples)

3. **Verify preprocessing matches training**:
   - Per-segment StandardScaler normalization
   - 16kHz sample rate
   - 40 mel bands
   - 100-frame segments

---

## Quick Reference: Full Pipeline

```bash
# === ON MAC ===
# 1. Generate calibration data
python3 generate_calibration_data.py --num-samples 50

# 2. Export to ONNX
python3 pytorch_to_hailo.py --pt wakeword_cnn.pt --onnx wakeword.onnx \
    --shape 1 1 40 100 --hw hailo8l --skip_hef

# 3. Transfer to Hailo PC
rsync -av calibration_data marius@192.168.178.62:~/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/
rsync -av wakeword.onnx marius@192.168.178.62:~/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/

# === ON HAILO COMPILATION PC ===
cd ~/Downloads/hailo8_ai_sw_suite_2025-10_docker
./hailo_ai_sw_suite_docker_run.sh --resume

# Inside Docker:
cd /local/shared_with_docker/
sudo chown -R hailo:ht calibration_data  # Fix permissions

hailo parser onnx wakeword.onnx --hw-arch hailo8l
hailo optimize wakeword.har --calib-set-path calibration_data/ --hw-arch hailo8l
hailo compiler wakeword_optimized.har --hw-arch hailo8l

# === BACK ON MAC ===
# 4. Transfer HEF back
rsync -avz marius@192.168.178.62:~/Downloads/hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/wakeword.hef ./

# 5. Deploy to Raspberry Pi
rsync -av wakeword.hef pi@192.168.178.194:/home/pi/WakeWordDetection/
git ls-files | rsync -av --files-from=- ./ pi@192.168.178.194:/home/pi/WakeWordDetection/

# === ON RASPBERRY PI ===
# 6. Run inference
python3 rpi_wakeword.py path/to/audio.wav
```

---

## Files Generated

| File | Size | Description |
|------|------|-------------|
| `wakeword.onnx` | ~550 KB | ONNX model (portable format) |
| `wakeword.har` | ~580 KB | Hailo Archive (parsed) |
| `wakeword_optimized.har` | ~3.5 MB | Optimized HAR (quantized) |
| `wakeword_compiled.har` | ~4.4 MB | Compiled HAR (with mappings) |
| `wakeword.hef` | ~877 KB | **Final Hailo Executable** |

Only `wakeword.hef` is needed for deployment on Raspberry Pi.

---

## Model Information

- **Architecture**: CNN (3 conv layers, 2 FC layers)
- **Input**: (1, 40, 100) NCHW → Hailo converts to (40, 100, 1) HWC
- **Output**: 2 classes (non-wake-word, wake-word)
- **Sample Rate**: 16 kHz
- **Features**: 40 mel-frequency bands
- **Window**: 100 frames (~1 second)
- **Normalization**: Per-segment StandardScaler (mean=0, std=1)

---

## Performance Metrics

After compilation, check the SNR (Signal-to-Noise Ratio) in the optimization output:

```
Output layers SNR: wakeword/output_layer1 SNR: 44.04 dB
```

**SNR Interpretation**:
- **> 40 dB**: Excellent quantization quality
- **30-40 dB**: Good quality
- **< 30 dB**: May need more calibration data or different optimization

Your model achieved **44.04 dB**, which indicates excellent quantization quality.

---

## Additional Resources

- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Hailo Dataflow Compiler User Guide](https://hailo.ai/developer-zone/documentation/dataflow-compiler/)
- [HailoRT Python API](https://hailo.ai/developer-zone/documentation/hailort/)

---

**Last Updated**: November 28, 2025
**Hailo DFC Version**: 3.33.0
**HailoRT Version**: 4.23.0
