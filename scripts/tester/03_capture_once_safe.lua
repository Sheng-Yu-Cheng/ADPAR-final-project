-- capture_once_safe.lua
-- Safe one-shot capture script.
--
-- This script DOES:
--   1. Start DCA1000 recording
--   2. Wait for DCA1000 to be ready
--   3. Trigger one finite radar frame
--   4. Wait for record completion
--
-- This script DOES NOT:
--   - Reset board
--   - PowerOff / PowerOn
--   - Download firmware
--   - StaticConfig
--   - RF Init
--   - DataConfig
--   - Profile / Chirp / FrameConfig
--
-- Assumptions:
--   1. AWR2243 is already connected and RF powered-up.
--   2. StaticConfig, DataConfig, ProfileConfig, ChirpConfig, FrameConfig already done.
--   3. DCA1000 setup already done.
--   4. No recording/framing is currently active.

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

adc_path = "C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\auto_capture.bin"

-- mmWave Studio / DCA1000 flow is safer with a gap before triggering.
pre_trigger_ms = 3000

-- Must be longer than actual finite-frame duration.
-- For your small 3Tx test:
--   16 frames * 40 ms = 0.64 s
-- so 3000 ms is enough.
-- For 64 frames:
--   64 frames * 40 ms = 2.56 s
-- use 5000~8000 ms.
post_trigger_ms = 3000

------------------------------------------------------------
-- CAPTURE
------------------------------------------------------------

WriteToLog("capture_once_safe.lua started\n", "green")

WriteToLog("Starting DCA1000 record...\n", "green")
ar1.CaptureCardConfig_StartRecord(adc_path, 1)

RSTD.Sleep(pre_trigger_ms)

WriteToLog("Triggering frame...\n", "green")
ar1.StartFrame()

RSTD.Sleep(post_trigger_ms)

WriteToLog("capture_once_safe.lua finished\n", "green")