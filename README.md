# Drone-Human Classification with TI AWR2243

This project uses a TI AWR2243BOOST radar module with DCA1000EVM raw ADC capture to display live range-Doppler heatmaps and classify the scene as `EMPTY`, `DRONE`, or `HUMAN`. The current main demo script is packet-loss-aware and tuned for a 60-second capture using 200 ms frame periods.

## Project Contents

- `scripts/radar/`: Lua setup scripts for mmWave Studio, DCA1000EVM, and AWR2243 configuration.
- `scripts/tester/`: Python radar receivers, visualizers, classifiers, packet sniffing helpers, and Tello drone control scripts.
- `figs/`: Figures used for the report or presentation.
- `slides.pptx`: Final presentation slides.
- `adc/`: Local raw ADC captures. This folder is large and is ignored by git; it is not included in the default submission zip.
- `requirement.txt`: Python package dependencies.
- `make_submission_zip.bat` / `make_submission_zip.ps1`: Windows packaging scripts for final submission.

## Hardware and Software

Required hardware:

- TI AWR2243BOOST
- TI DCA1000EVM
- CAT6 Ethernet cable
- Power adapters for the radar boards
- DJI/Ryze Tello drone, only needed for the drone-control demo

Required software:

- Windows
- Python 3.10 or newer
- TI mmWave Studio `03.00.00.14`
- Npcap or WinPcap if using Scapy packet sniffing utilities

## Hardware Setup

1. Connect the DCA1000EVM and AWR2243BOOST according to the TI mmWave Studio official guide.
2. Connect the CAT6 Ethernet cable between the DCA1000EVM and the PC.
3. Power on the boards. Connect the power adapters before starting the capture workflow.
4. Configure the PC Ethernet interface for the DCA1000 network. The Python scripts currently bind to:

```text
PC_IP  = 192.168.33.30
DCA_IP = 192.168.33.180
DATA_PORT = 4098
```

If your Ethernet adapter uses a different IP, update the `PC_IP` constant in the Python script before running the demo.

## Python Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirement.txt
```

## Radar Setup in mmWave Studio

1. Open mmWave Studio.
2. Connect to the DCA1000EVM.
3. Connect to the AWR2243.
4. Run the radar setup scripts in order:

```text
scripts/radar/01_dca_1000_evm_setup.lua
scripts/radar/02_awr2243_connection_setup.lua
scripts/radar/03_static_config_setup.lua
scripts/radar/04_data_config_setup.lua
scripts/radar/05_sensor_config_setup.lua
```

The live classifier expects the radar frame configuration to match:

```text
FrameConfig(0, 2, 300, 64, 200, 0, 1)
```

This corresponds to 300 frames, 64 loops, and a 200 ms frame period, for a 60-second capture.

## Run the Main Live Classifier

Start the Python receiver before starting the frame capture in mmWave Studio:

```powershell
python scripts/tester/live_event_classifier_v5b_packet_loss_safe.py
```

Then start frame capture in mmWave Studio. The Python window should show a live range-Doppler heatmap and classification output for `EMPTY`, `DRONE`, or `HUMAN`.

Notes:

- Keep the scene empty during the first background-calibration frames.
- The main script writes a raw backup into `adc/` by default.
- If the heatmap is too dark or saturated, tune `FIXED_HEATMAP_VMIN_DB` and `FIXED_HEATMAP_VMAX_DB`.
- If the classifier misses slow motion or drone hover, tune `MIN_ABS_VELOCITY_MPS` and the event-classification thresholds near the top of the script.

## Drone Control Demo

Connect the PC to the Tello Wi-Fi network, then run:

```powershell
python scripts/tester/drone-control.py
```

The drone-control scripts require `djitellopy` and `pygame`, which are included in `requirement.txt`.

## Useful Scripts

- `scripts/tester/live_event_classifier_v5b_packet_loss_safe.py`: Main final demo classifier with packet-loss-safe raw backup.
- `scripts/tester/live_event_classifier_v5b_microdoppler_rotor_60s_200ms.py`: Earlier classifier variant with micro-Doppler/rotor-oriented features.
- `scripts/tester/dca1000_live_stream_plot.py`: Basic live stream plotter.
- `scripts/tester/dca1000_socket_receiver.py`: Raw DCA1000 UDP receiver.
- `scripts/tester/list_scapy_ifaces.py`: Lists Windows interfaces for Scapy.
- `scripts/tester/udp_sniff.py`: UDP sniffing helper.
- `scripts/tester/drone-control.py`: Tello keyboard-control demo.

## Build the Submission Zip

Double-click:

```text
make_submission_zip.bat
```

or run:

```powershell
.\make_submission_zip.ps1
```

The zip will be created under `dist/`. By default, the package includes the source code, radar Lua scripts, figures, slides, README, and dependency file. It excludes `.git`, `.venv`, `dist`, Python cache folders, temporary Office files, and the large `adc/` raw-data folder.

To include `adc/` raw captures:

```powershell
.\make_submission_zip.ps1 -IncludeAdc
```

Only include raw ADC data if the submission rule requires it, because the files are very large.

## Troubleshooting

- If the Python script cannot bind the UDP socket, check that the PC Ethernet IP matches `PC_IP`.
- If no packets arrive, confirm DCA1000 connection and start frame capture after the Python receiver is already running.
- If Scapy cannot capture packets, install Npcap and run the terminal as Administrator.
- If the Tello does not respond, reconnect to the Tello Wi-Fi and make sure no other app is controlling the drone.
