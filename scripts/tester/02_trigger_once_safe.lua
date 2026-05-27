WriteToLog("trigger_once_safe.lua started\n", "green")

-- Assumption:
-- Radar is already connected, RF powered-up, StaticConfig/DataConfig/SensorConfig already done.
-- This script does NOT reset, does NOT power off, does NOT download firmware.

ar1.StartFrame()

WriteToLog("trigger_once_safe.lua finished\n", "green")