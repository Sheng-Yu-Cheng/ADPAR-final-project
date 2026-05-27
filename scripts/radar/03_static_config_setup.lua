-- setup_static_config_safe.lua
-- Safe StaticConfig setup for AWR2243BOOST in mmWave Studio.
--
-- This script does NOT:
--   - Reset board
--   - PowerOff / PowerOn
--   - Download firmware
--   - StartFrame
--   - Configure DCA1000
--   - Change Profile / Chirp / Frame
--
-- Assumptions:
--   - RS232 is connected
--   - Firmware already loaded
--   - PowerOn / RfEnable already passed
--   - No frame is currently running
--   - No DCA1000 recording is currently running

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

-- Choose number of TX antennas to enable:
--   1 => Tx0 only
--   2 => Tx0 + Tx1
--   3 => Tx0 + Tx1 + Tx2, for TDM-MIMO
NUM_TX = 3

-- Usually keep all 4 RX enabled.
ENABLE_RX0 = 1
ENABLE_RX1 = 1
ENABLE_RX2 = 1
ENABLE_RX3 = 1

-- ADC output config used in your successful setup.
-- From your previous successful log:
-- ar1.ChanNAdcConfig(..., 2, 1, 0)
-- 2 = Complex1x
-- 1 = 16-bit / ADC output setting used by mmWave Studio GUI
-- 0 = I First / no IQ swap setting used by your GUI
ADC_FORMAT = 2
ADC_BITS_OR_OUT_FMT = 1
IQ_SWAP = 0

-- Frequency limits.
FREQ_LOW_GHZ = 77
FREQ_HIGH_GHZ = 81

-- LP ADC mode:
-- ar1.LPModConfig(0, 0) matched your successful setup.
LP_ADC_MODE = 0
LP_RESERVED = 0

-- For 3Tx on AWR2243BOOST, TI says RF LDO bypass + PALDO disable is required.
-- This maps to ar1.RfLdoBypassConfig(0x3).
AUTO_SET_LDO_BYPASS_FOR_3TX = true

-- For 1Tx/2Tx, do NOT automatically disable LDO bypass.
-- Reason: if a previous safe 3Tx config already set it, toggling it mid-session is unnecessary.
-- If you want deterministic 1Tx/2Tx behavior after fresh power-up, manually decide later.
TOUCH_LDO_CONFIG_FOR_1_2TX = false

-- Whether to run RF Init at the end.
-- First test can use false to verify only config APIs.
-- For normal setup, set this to true after you trust the script.
RUN_RF_INIT = true

-- Small delay after each API.
API_SLEEP_MS = 200

------------------------------------------------------------
-- INTERNAL HELPERS
------------------------------------------------------------

function log(msg, color)
    if color == nil then
        color = "green"
    end
    WriteToLog(msg .. "\n", color)
end

function die(msg)
    WriteToLog("ERROR: " .. msg .. "\n", "red")
    error(msg)
end

function sleep()
    RSTD.Sleep(API_SLEEP_MS)
end

------------------------------------------------------------
-- VALIDATION
------------------------------------------------------------

if NUM_TX ~= 1 and NUM_TX ~= 2 and NUM_TX ~= 3 then
    die("NUM_TX must be 1, 2, or 3.")
end

local tx0 = 1
local tx1 = 0
local tx2 = 0

if NUM_TX >= 2 then
    tx1 = 1
end

if NUM_TX >= 3 then
    tx2 = 1
end

log("====================================================", "yellow")
log("setup_static_config_safe.lua started", "yellow")
log(string.format("Requested NUM_TX = %d", NUM_TX), "yellow")
log(string.format("TX enable: Tx0=%d, Tx1=%d, Tx2=%d", tx0, tx1, tx2), "yellow")
log(string.format("RX enable: Rx0=%d, Rx1=%d, Rx2=%d, Rx3=%d", ENABLE_RX0, ENABLE_RX1, ENABLE_RX2, ENABLE_RX3), "yellow")
log("This script will not reset, power-cycle, download firmware, or start frame.", "yellow")
log("====================================================", "yellow")

------------------------------------------------------------
-- STATIC CONFIG: CHANNEL + ADC
------------------------------------------------------------

log("Step 1: Channel + ADC config")

-- Known-good call pattern from your logs:
-- 1Tx:
--   ar1.ChanNAdcConfig(1, 0, 0, 1, 1, 1, 1, 2, 1, 0)
-- 3Tx:
--   ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)

ar1.ChanNAdcConfig(
    tx0, tx1, tx2,
    ENABLE_RX0, ENABLE_RX1, ENABLE_RX2, ENABLE_RX3,
    ADC_FORMAT,
    ADC_BITS_OR_OUT_FMT,
    IQ_SWAP
)

sleep()

------------------------------------------------------------
-- STATIC CONFIG: RF LDO BYPASS
------------------------------------------------------------

if NUM_TX == 3 then
    if AUTO_SET_LDO_BYPASS_FOR_3TX then
        log("Step 2: NUM_TX=3, applying RF LDO bypass config 0x3", "yellow")
        log("Reason: AWR2243BOOST requires RF LDO Bypass + PALDO I/P Disable for third TX.", "yellow")
        ar1.RfLdoBypassConfig(0x3)
        sleep()
    else
        die("NUM_TX=3 but AUTO_SET_LDO_BYPASS_FOR_3TX is false. Refusing for safety.")
    end
else
    if TOUCH_LDO_CONFIG_FOR_1_2TX then
        log("Step 2: NUM_TX<3 and TOUCH_LDO_CONFIG_FOR_1_2TX=true, setting RfLdoBypassConfig(0x0)", "yellow")
        ar1.RfLdoBypassConfig(0x0)
        sleep()
    else
        log("Step 2: NUM_TX<3, leaving RF LDO bypass config unchanged.", "yellow")
    end
end

------------------------------------------------------------
-- STATIC CONFIG: LP MODE
------------------------------------------------------------

log("Step 3: LP ADC mode config")
ar1.LPModConfig(LP_ADC_MODE, LP_RESERVED)
sleep()

------------------------------------------------------------
-- STATIC CONFIG: FREQUENCY LIMITS
------------------------------------------------------------

log("Step 4: Cal monitor frequency limit config")
ar1.SetCalMonFreqLimitConfig(FREQ_LOW_GHZ, FREQ_HIGH_GHZ)
sleep()

log("Step 5: TX power monitor frequency limit config")

-- Your successful manual call:
-- ar1.RfSetCalMonFreqTxPowLimitConfig(77, 77, 77, 81, 81, 81, 0, 0, 0)
ar1.RfSetCalMonFreqTxPowLimitConfig(
    FREQ_LOW_GHZ, FREQ_LOW_GHZ, FREQ_LOW_GHZ,
    FREQ_HIGH_GHZ, FREQ_HIGH_GHZ, FREQ_HIGH_GHZ,
    0, 0, 0
)

sleep()

------------------------------------------------------------
-- OPTIONAL RF INIT
------------------------------------------------------------

if RUN_RF_INIT then
    log("Step 6: RF Init", "yellow")
    ar1.RfInit()
    sleep()
else
    log("Step 6: RF Init skipped because RUN_RF_INIT=false", "yellow")
end

log("====================================================", "green")
log("setup_static_config_safe.lua finished", "green")
log("Next safe step: DataConfig, then SensorConfig Profile/Chirp/Frame.", "green")
log("====================================================", "green")