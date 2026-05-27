-- sensor_config_3tx_tdm_safe.lua
-- Safe SensorConfig for AWR2243BOOST 3Tx x 4Rx TDM-MIMO.
--
-- This script DOES:
--   - Configure one profile
--   - Configure 3 chirps:
--       chirp 0 -> Tx0
--       chirp 1 -> Tx1
--       chirp 2 -> Tx2
--   - Configure finite frame
--
-- This script DOES NOT:
--   - Reset board
--   - PowerOff / PowerOn
--   - Download firmware
--   - StaticConfig
--   - RF Init
--   - DataConfig
--   - DCA1000 record
--   - StartFrame
--
-- Assumptions:
--   - connect_awr2243_safe.lua already passed
--   - setup_static_config_safe.lua already passed with NUM_TX = 3
--   - RF LDO Bypass was enabled via ar1.RfLdoBypassConfig(0x3)
--   - data_config_safe.lua already passed
--   - DCA1000 setup may be done, but recording is not active
--   - No frame is currently running

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

-- Profile parameters, copied from your successful manual log:
-- ar1.ProfileConfig(0, 77, 100, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 10000, 0, 0, 94)

PROFILE_ID = 0
START_FREQ_GHZ = 77
IDLE_TIME_US = 100
ADC_START_TIME_US = 6
RAMP_END_TIME_US = 60

TX_OUT_POWER_BACKOFF_CODE = 0
TX_PHASE_SHIFTER = 0
FREQ_SLOPE_CONST_MHZ_PER_US = 29.982
TX_START_TIME_US = 0
NUM_ADC_SAMPLES = 256
DIG_OUT_SAMPLE_RATE_KSPS = 10000
HPF_CORNER_FREQ1 = 0
HPF_CORNER_FREQ2 = 0
RX_GAIN_CODE = 94

-- TDM-MIMO chirp indices
CHIRP_TX0 = 0
CHIRP_TX1 = 1
CHIRP_TX2 = 2

-- First safe test frame settings from your successful manual log:
-- ar1.FrameConfig(0, 2, 16, 64, 40, 0, 1)
START_CHIRP = 0
END_CHIRP = 2
NUM_FRAMES = 16
NUM_CHIRP_LOOPS = 64
FRAME_PERIODICITY_MS = 40
TRIGGER_DELAY_US = 0
TRIGGER_SELECT = 1  -- Software trigger

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

function sleep()
    RSTD.Sleep(API_SLEEP_MS)
end

------------------------------------------------------------
-- START
------------------------------------------------------------

log("====================================================", "yellow")
log("sensor_config_3tx_tdm_safe.lua started", "yellow")
log("This script configures Profile + 3Tx TDM chirps + Frame.", "yellow")
log("It will NOT start record or trigger frame.", "yellow")
log("TDM-MIMO chirps:", "yellow")
log("  chirp 0 -> Tx0", "yellow")
log("  chirp 1 -> Tx1", "yellow")
log("  chirp 2 -> Tx2", "yellow")
log("====================================================", "yellow")

------------------------------------------------------------
-- PROFILE CONFIG
------------------------------------------------------------

log("Step 1: ProfileConfig", "yellow")

ar1.ProfileConfig(
    PROFILE_ID,
    START_FREQ_GHZ,
    IDLE_TIME_US,
    ADC_START_TIME_US,
    RAMP_END_TIME_US,
    TX_OUT_POWER_BACKOFF_CODE,
    TX_OUT_POWER_BACKOFF_CODE,
    TX_OUT_POWER_BACKOFF_CODE,
    TX_PHASE_SHIFTER,
    TX_PHASE_SHIFTER,
    TX_PHASE_SHIFTER,
    FREQ_SLOPE_CONST_MHZ_PER_US,
    TX_START_TIME_US,
    NUM_ADC_SAMPLES,
    DIG_OUT_SAMPLE_RATE_KSPS,
    HPF_CORNER_FREQ1,
    HPF_CORNER_FREQ2,
    RX_GAIN_CODE
)

sleep()

------------------------------------------------------------
-- CHIRP CONFIGS
------------------------------------------------------------

log("Step 2: ChirpConfig chirp 0 -> Tx0", "yellow")

-- ar1.ChirpConfig(startIdx, endIdx, profileId,
--                 startFreqVar, freqSlopeVar, idleTimeVar, adcStartTimeVar,
--                 tx0Enable, tx1Enable, tx2Enable)

ar1.ChirpConfig(
    CHIRP_TX0,
    CHIRP_TX0,
    PROFILE_ID,
    0,
    0,
    0,
    0,
    1,
    0,
    0
)

sleep()

log("Step 3: ChirpConfig chirp 1 -> Tx1", "yellow")

ar1.ChirpConfig(
    CHIRP_TX1,
    CHIRP_TX1,
    PROFILE_ID,
    0,
    0,
    0,
    0,
    0,
    1,
    0
)

sleep()

log("Step 4: ChirpConfig chirp 2 -> Tx2", "yellow")

ar1.ChirpConfig(
    CHIRP_TX2,
    CHIRP_TX2,
    PROFILE_ID,
    0,
    0,
    0,
    0,
    0,
    0,
    1
)

sleep()

------------------------------------------------------------
-- TEST SOURCE OFF
------------------------------------------------------------

log("Step 5: Disable test source", "yellow")
ar1.DisableTestSource(0)
sleep()

------------------------------------------------------------
-- FRAME CONFIG
------------------------------------------------------------

log("Step 6: FrameConfig", "yellow")
log(string.format("Start chirp = %d, End chirp = %d", START_CHIRP, END_CHIRP), "yellow")
log(string.format("Frames = %d, Chirp loops = %d", NUM_FRAMES, NUM_CHIRP_LOOPS), "yellow")
log(string.format("Frame periodicity = %d ms", FRAME_PERIODICITY_MS), "yellow")

ar1.FrameConfig(
    START_CHIRP,
    END_CHIRP,
    NUM_FRAMES,
    NUM_CHIRP_LOOPS,
    FRAME_PERIODICITY_MS,
    TRIGGER_DELAY_US,
    TRIGGER_SELECT
)

sleep()

------------------------------------------------------------
-- DONE
------------------------------------------------------------

log("====================================================", "green")
log("sensor_config_3tx_tdm_safe.lua finished", "green")
log("Expected file size for this config:", "green")
log("  256 samples * 4 RX * 2 IQ * 2 bytes * 64 loops * 3 chirps * 16 frames", "green")
log("  = 12,582,912 bytes", "green")
log("Next safe step: DCA1000 ARM / StartRecord, wait, then Trigger Frame.", "green")
log("====================================================", "green")

