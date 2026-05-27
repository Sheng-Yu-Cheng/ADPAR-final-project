-- connect_awr2243_safe.lua
-- Safe connection script for AWR2243BOOST + DCA1000EVM using mmWave Studio 03.00.00.14
--
-- This script DOES:
--   1. Select single-chip radar mode
--   2. Optionally FullReset + SOPControl(2)
--   3. Connect RS232
--   4. Select XWR2243 / 77G
--   5. Download BSS/MSS firmware
--   6. PowerOn / RF Enable
--   7. Print firmware versions
--
-- This script DOES NOT:
--   - PowerOff
--   - Flash / erase serial flash
--   - Use metaImage
--   - Configure StaticConfig
--   - RF Init
--   - Configure Profile/Chirp/Frame
--   - StartFrame
--   - Configure or arm DCA1000

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

-- From your successful log:
-- ar1.Connect(20, 921600, 1000)
COM_PORT = 20

BAUD_RATE = 921600
CONNECT_TIMEOUT_MS = 1000

-- SOP 2 = development mode, the one you used successfully.
SOP_MODE = 2

-- Set true only at the beginning of a fresh setup.
-- Do NOT run this while capture/framing is active.
DO_FULL_RESET = true

-- Firmware paths from your successful setup.
BSS_FW_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\radarss\\xwr22xx_radarss.bin"
MSS_FW_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\masterss\\xwr22xx_masterss.bin"

-- Set false only if firmware is already loaded in the same session.
DOWNLOAD_FIRMWARE = true

-- Set true for the normal connection flow.
DO_POWER_ON_AND_RF_ENABLE = true

API_SLEEP_MS = 300

------------------------------------------------------------
-- HELPERS
------------------------------------------------------------

function log(msg, color)
    if color == nil then
        color = "green"
    end
    WriteToLog(msg .. "\n", color)
end

function warn(msg)
    WriteToLog("WARNING: " .. msg .. "\n", "yellow")
end

function fail(msg)
    WriteToLog("ERROR: " .. msg .. "\n", "red")
    error(msg)
end

function sleep_ms(ms)
    RSTD.Sleep(ms)
end

function sleep()
    RSTD.Sleep(API_SLEEP_MS)
end

------------------------------------------------------------
-- START
------------------------------------------------------------

log("====================================================", "yellow")
log("connect_awr2243_safe.lua started", "yellow")
log(string.format("COM_PORT = %d", COM_PORT), "yellow")
log(string.format("BAUD_RATE = %d", BAUD_RATE), "yellow")
log(string.format("DO_FULL_RESET = %s", tostring(DO_FULL_RESET)), "yellow")
log(string.format("DOWNLOAD_FIRMWARE = %s", tostring(DOWNLOAD_FIRMWARE)), "yellow")
log(string.format("DO_POWER_ON_AND_RF_ENABLE = %s", tostring(DO_POWER_ON_AND_RF_ENABLE)), "yellow")
log("This script will NOT configure chirps, start frames, erase flash, or PowerOff.", "yellow")
log("====================================================", "yellow")

------------------------------------------------------------
-- RADAR MODE
------------------------------------------------------------

log("Step 1: Select single-chip radar mode")
ar1.selectRadarMode(0)
sleep()

log("Step 2: Select non-cascade mode")
ar1.selectCascadeMode(0)
sleep()

------------------------------------------------------------
-- OPTIONAL RESET + SOP2
------------------------------------------------------------

if DO_FULL_RESET then
    warn("Doing FullReset + SOPControl(2). Use this only before configuration/capture starts.")

    log("Opening GPIO control port")
    ar1.OpenGpioControlPort()
    sleep()

    log("Opening board control port")
    ar1.OpenBoardControlPort()
    sleep()

    log("FullReset")
    ar1.FullReset()
    sleep_ms(800)

    log("Closing board control port")
    ar1.CloseBoardControlPort()
    sleep()

    log("Closing GPIO control port")
    ar1.CloseGpioControlPort()
    sleep()

    log("Setting SOP mode = 2")
    ar1.SOPControl(SOP_MODE)
    sleep_ms(1000)
else
    warn("Skipping FullReset/SOPControl. Only do this if board is already in SOP2 and clean state.")
end

------------------------------------------------------------
-- RS232 CONNECT
------------------------------------------------------------

log(string.format("Step 3: Connect RS232 on COM%d", COM_PORT))
ar1.Connect(COM_PORT, BAUD_RATE, CONNECT_TIMEOUT_MS)
sleep_ms(1500)

log("Check connection")
ar1.Calling_IsConnected()
sleep()

------------------------------------------------------------
-- DEVICE SELECTION
------------------------------------------------------------

log("Step 4: Select chip/version/device variant")

-- Your successful log includes AR1243 selection first, then XWR2243.
ar1.SelectChipVersion("AR1243")
sleep()

ar1.SelectChipVersion("AR1243")
sleep()

ar1.deviceVariantSelection("XWR2243")
sleep()

ar1.frequencyBandSelection("77G")
sleep()

ar1.SelectChipVersion("XWR2243")
sleep()

log("Device should now report: XWR2243/ASIL-B/SOP:2/ES:1.1", "yellow")

------------------------------------------------------------
-- FIRMWARE DOWNLOAD
------------------------------------------------------------

if DOWNLOAD_FIRMWARE then
    log("Step 5: Download BSS firmware", "yellow")
    log(BSS_FW_PATH, "yellow")
    ar1.DownloadBSSFw(BSS_FW_PATH)
    sleep_ms(1200)

    log("Read BSS firmware version")
    ar1.GetBSSFwVersion()
    sleep()

    log("Read BSS patch firmware version")
    ar1.GetBSSPatchFwVersion()
    sleep()

    log("Step 6: Download MSS firmware", "yellow")
    log(MSS_FW_PATH, "yellow")
    ar1.DownloadMSSFw(MSS_FW_PATH)
    sleep_ms(2000)

    log("Read MSS firmware version")
    ar1.GetMSSFwVersion()
    sleep()
else
    warn("Skipping firmware download. This is only safe if firmware is already loaded in this session.")
end

------------------------------------------------------------
-- POWER ON + RF ENABLE
------------------------------------------------------------

if DO_POWER_ON_AND_RF_ENABLE then
    log("Step 7: MSS PowerOn", "yellow")
    ar1.PowerOn(0, 1000, 0, 0)
    sleep_ms(2500)

    log("Re-select chip version after PowerOn")
    ar1.SelectChipVersion("AR1243")
    sleep()

    ar1.SelectChipVersion("XWR2243")
    sleep()

    log("Step 8: RF Enable", "yellow")
    ar1.RfEnable()
    sleep_ms(1500)

    log("Read final firmware versions")
    ar1.GetMSSFwVersion()
    sleep()

    ar1.GetBSSFwVersion()
    sleep()

    ar1.GetBSSPatchFwVersion()
    sleep()
else
    warn("Skipping PowerOn/RfEnable.")
end

------------------------------------------------------------
-- DONE
------------------------------------------------------------

log("====================================================", "green")
log("connect_awr2243_safe.lua finished", "green")
log("Next safe step: run setup_static_config_safe.lua, then DataConfig, then SensorConfig.", "green")
log("====================================================", "green")