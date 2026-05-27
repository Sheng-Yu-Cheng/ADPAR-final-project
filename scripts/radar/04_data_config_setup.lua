-- data_config_safe.lua
-- Safe DataConfig setup for AWR2243BOOST + DCA1000EVM in mmWave Studio.
--
-- This script DOES:
--   - Configure LVDS data path
--   - Configure LVDS clock
--   - Configure LVDS lane format
--
-- This script DOES NOT:
--   - Reset board
--   - PowerOff / PowerOn
--   - Download firmware
--   - StaticConfig
--   - RF Init
--   - Profile / Chirp / Frame
--   - DCA1000 record
--   - StartFrame
--
-- Assumptions:
--   - Connection script already passed
--   - StaticConfig already passed
--   - RF Init already passed
--   - No frame is currently running
--   - No DCA1000 recording is currently running

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

-- These are exactly from your successful log:
-- ar1.DataPathConfig(513, 1216644097, 0)
-- ar1.LvdsClkConfig(1, 1)
-- ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)

DATA_PATH_CONFIG_ARG0 = 513
DATA_PATH_CONFIG_ARG1 = 1216644097
DATA_PATH_CONFIG_ARG2 = 0

LVDS_CLK_ARG0 = 1
LVDS_CLK_ARG1 = 1

LANE_FORMAT = 0
LANE1_ENABLE = 1
LANE2_ENABLE = 1
LANE3_ENABLE = 1
LANE4_ENABLE = 1
MSB_FIRST = 1
CRC_ENABLE = 0
PACKET_END_PULSE_ENABLE = 0

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
log("data_config_safe.lua started", "yellow")
log("This script only configures DataConfig / LVDS output.", "yellow")
log("It will NOT reset, RF init, configure chirps, or start frames.", "yellow")
log("====================================================", "yellow")

------------------------------------------------------------
-- DATA PATH CONFIG
------------------------------------------------------------

log("Step 1: DataPathConfig")
log("Expected GUI equivalent: LVDS, ADC_ONLY, Packet1 suppress, CQ 16-bit", "yellow")

ar1.DataPathConfig(
    DATA_PATH_CONFIG_ARG0,
    DATA_PATH_CONFIG_ARG1,
    DATA_PATH_CONFIG_ARG2
)

sleep()

------------------------------------------------------------
-- LVDS CLOCK CONFIG
------------------------------------------------------------

log("Step 2: LvdsClkConfig")
log("Expected GUI equivalent: DDR Clock, 600 Mbps", "yellow")

ar1.LvdsClkConfig(
    LVDS_CLK_ARG0,
    LVDS_CLK_ARG1
)

sleep()

------------------------------------------------------------
-- LVDS LANE CONFIG
------------------------------------------------------------

log("Step 3: LVDSLaneConfig")
log("Expected GUI equivalent: Format 0, Lane1~4 enabled, MSB first, CRC off", "yellow")

ar1.LVDSLaneConfig(
    LANE_FORMAT,
    LANE1_ENABLE,
    LANE2_ENABLE,
    LANE3_ENABLE,
    LANE4_ENABLE,
    MSB_FIRST,
    CRC_ENABLE,
    PACKET_END_PULSE_ENABLE
)

sleep()

------------------------------------------------------------
-- DONE
------------------------------------------------------------

log("====================================================", "green")
log("data_config_safe.lua finished", "green")
log("Next safe step: SensorConfig Profile / Chirp / Frame.", "green")
log("====================================================", "green")