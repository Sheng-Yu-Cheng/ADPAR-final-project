-- dca1000_setup_safe.lua
-- Safe DCA1000 setup script for mmWave Studio.
--
-- This script DOES:
--   - Read DCA1000 DLL version
--   - Select DCA1000 capture device
--   - Configure Ethernet
--   - Configure capture mode
--   - Configure packet delay
--   - Read FPGA version
--
-- This script DOES NOT:
--   - Reset radar
--   - Power off radar
--   - Download firmware
--   - StaticConfig / RF Init
--   - Profile / Chirp / Frame
--   - Start record
--   - Start frame

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

PC_IP = "192.168.33.30"
DCA1000_IP = "192.168.33.180"
DCA1000_MAC = "12:34:56:78:90:12"

CONFIG_PORT = 4096
DATA_PORT = 4098

-- From your successful manual log:
-- ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)
MODE_ARG0 = 1
MODE_ARG1 = 1
MODE_ARG2 = 1
MODE_ARG3 = 2
MODE_ARG4 = 3
MODE_ARG5 = 30

PACKET_DELAY_US = 25

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
log("dca1000_setup_safe.lua started", "yellow")
log("This script only configures DCA1000.", "yellow")
log("It will NOT start record or trigger radar frame.", "yellow")
log("====================================================", "yellow")

------------------------------------------------------------
-- DCA1000 SETUP
------------------------------------------------------------

log("Step 1: Get DCA1000 DLL version", "yellow")
ar1.GetCaptureCardDllVersion()
sleep()

log("Step 2: Select DCA1000 capture device", "yellow")
ar1.SelectCaptureDevice("DCA1000")
sleep()

log("Step 3: Configure DCA1000 Ethernet", "yellow")
log("PC IP      = " .. PC_IP, "yellow")
log("DCA1000 IP = " .. DCA1000_IP, "yellow")
log("Config port = " .. tostring(CONFIG_PORT) .. ", Data port = " .. tostring(DATA_PORT), "yellow")

ar1.CaptureCardConfig_EthInit(
    PC_IP,
    DCA1000_IP,
    DCA1000_MAC,
    CONFIG_PORT,
    DATA_PORT
)

sleep()

log("Step 4: Configure DCA1000 capture mode", "yellow")

ar1.CaptureCardConfig_Mode(
    MODE_ARG0,
    MODE_ARG1,
    MODE_ARG2,
    MODE_ARG3,
    MODE_ARG4,
    MODE_ARG5
)

sleep()

log("Step 5: Configure DCA1000 packet delay", "yellow")
ar1.CaptureCardConfig_PacketDelay(PACKET_DELAY_US)
sleep()

log("Step 6: Read DCA1000 FPGA version", "yellow")
ar1.GetCaptureCardFPGAVersion()
sleep()

------------------------------------------------------------
-- DONE
------------------------------------------------------------

log("====================================================", "green")
log("dca1000_setup_safe.lua finished", "green")
log("Expected successful messages:", "green")
log("  FPGA Configuration command : Success", "green")
log("  Configure Record command : Success", "green")
log("  FPGA Version : 2.9 [Record]", "green")
log("====================================================", "green")