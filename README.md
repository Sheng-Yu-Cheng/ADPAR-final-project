# Drone-Human Classification with TI AWR2243

Final project submission for real-time drone/human/empty-scene classification using TI AWR2243BOOST radar, DCA1000EVM raw ADC streaming, and a DJI/Ryze Tello drone demo.

## Submission Files

The final submission package contains only:

- `main.py`: main radar live classifier, copied from `scripts/tester/live_event_classifier_v5b_packet_loss_safe.py`
- `drone-control.py`: Tello drone keyboard-control demo
- `requirement.txt`: Python dependencies
- `slides.pdf`: final presentation slides
- `README.md`: setup and execution guide
- `lua/`: mmWave Studio Lua setup scripts

## Hardware

- TI AWR2243BOOST
- TI DCA1000EVM
- CAT6 Ethernet cable
- Power adapters for the radar boards
- DJI/Ryze Tello drone, only needed for `drone-control.py`

## Software

- Windows
- Python 3.10 or newer
- TI mmWave Studio `03.00.00.14`
- Npcap or WinPcap, only needed for Scapy packet-sniffing helpers

## Python Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirement.txt
```

## Radar Network Settings

The main classifier currently expects the DCA1000 network to use:

```text
PC_IP    = 192.168.33.30
DCA_IP   = 192.168.33.180
DATA_PORT = 4098
```

If your PC Ethernet adapter uses a different IP, update the `PC_IP` constant near the top of `main.py`.

## mmWave Studio Setup

1. Connect the AWR2243BOOST and DCA1000EVM according to the TI mmWave Studio guide.
2. Connect the DCA1000EVM to the PC using a CAT6 Ethernet cable.
3. Power on the boards.
4. Open mmWave Studio.
5. Run the Lua scripts in `lua/` in order:

```text
lua/01_dca_1000_evm_setup.lua
lua/02_awr2243_connection_setup.lua
lua/03_static_config_setup.lua
lua/04_data_config_setup.lua
lua/05_sensor_config_setup.lua
```

The classifier expects this frame configuration:

```text
FrameConfig(0, 2, 300, 64, 200, 0, 1)
```

This means 300 frames, 64 loops, and a 200 ms frame period, for a 60-second capture.

## Run the Radar Classifier

Start the Python receiver first:

```powershell
python main.py
```

Then start frame capture in mmWave Studio. A live range-Doppler heatmap should appear, and the program will classify the scene as:

```text
EMPTY
DRONE
HUMAN
```

Notes:

- Keep the scene empty during the first background-calibration frames.
- `main.py` writes a raw backup into `adc/` by default when run from the full project folder.
- If the heatmap is too dark or saturated, tune `FIXED_HEATMAP_VMIN_DB` and `FIXED_HEATMAP_VMAX_DB`.
- If the classifier misses slow motion or drone hover, tune `MIN_ABS_VELOCITY_MPS` and the event-classification thresholds near the top of `main.py`.

## Run the Drone Control Demo

Connect the PC to the Tello Wi-Fi network, then run:

```powershell
python drone-control.py
```

This script uses `djitellopy` and `pygame`, both included in `requirement.txt`.

## Build the Submission Zip

From the full project folder, run:

```powershell
.\scripts\util\make_submission_zip.ps1
```

The script refreshes `dist/submission/` with the required files and creates `dist/ADPAR-final-project-submission.zip`.

## Troubleshooting

- If `main.py` cannot bind the UDP socket, confirm the Ethernet adapter IP matches `PC_IP`.
- If no radar packets arrive, start `main.py` before starting frame capture in mmWave Studio.
- If the Tello does not respond, reconnect to the Tello Wi-Fi and make sure no other app is controlling the drone.
