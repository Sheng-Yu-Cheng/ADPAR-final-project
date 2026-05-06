WriteToLog("capture_once_safe.lua started\n", "green")

-- Safe assumptions:
-- 1. AWR2243 is already connected and RF powered-up.
-- 2. StaticConfig, DataConfig, ProfileConfig, ChirpConfig, FrameConfig already done.
-- 3. DCA1000 setup already done and FPGA version can be read.
-- 4. This script does NOT reset or power-cycle the radar.

adc_path = "C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\auto_capture.bin"

WriteToLog("Starting DCA1000 record...\n", "green")
ar1.CaptureCardConfig_StartRecord(adc_path, 1)

-- TI docs recommend a gap between DCA1000 ARM/start record and Trigger Frame.
-- You previously succeeded with about 5 seconds. Use 3000 ms first.
RSTD.Sleep(3000)

WriteToLog("Triggering frame...\n", "green")
ar1.StartFrame()

-- Wait for finite frame and DCA1000 record completion.
-- Your previous frame was 8 frames * 40 ms = 0.32 s.
-- Use 3000 ms first to be safe.
RSTD.Sleep(3000)

WriteToLog("capture_once_safe.lua finished\n", "green")