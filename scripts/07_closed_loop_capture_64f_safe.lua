WriteToLog("closed_loop_capture_64f_safe.lua started\n", "green")

-- SAFE SCRIPT:
-- Does NOT reset, power off, download firmware, RF init, or change radar config.
-- Assumes:
--   1. AWR2243 is already connected.
--   2. SPI is connected.
--   3. RF is powered-up.
--   4. StaticConfig/DataConfig/ProfileConfig/ChirpConfig already passed.
--   5. FrameConfig has already been manually set to 64 frames.
--   6. DCA1000 setup is already working.

base_dir = "C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\"
decision_path = base_dir .. "latest_decision.txt"

num_scans = 10

-- 64 frames * 40 ms = 2.56 sec actual radar data.
-- Keep conservative waits first.
pre_trigger_ms = 3000
post_trigger_ms = 8000

function read_decision()
    local f = io.open(decision_path, "r")
    if f == nil then
        return "NO_DECISION"
    end

    local line = f:read("*line")
    f:close()

    if line == nil then
        return "EMPTY_DECISION_FILE"
    end

    return line
end

for i = 1, num_scans do
    local prev_decision = read_decision()
    WriteToLog(string.format("Before scan %d/%d, previous decision = %s\n", i, num_scans, prev_decision), "yellow")

    if prev_decision == "STOP" then
        WriteToLog("STOP decision received. Exiting loop safely.\n", "red")
        break
    end

    local adc_path = base_dir .. string.format("cl64_%03d.bin", i)

    WriteToLog(string.format("Scan %d/%d: start DCA1000 record -> %s\n", i, num_scans, adc_path), "green")
    ar1.CaptureCardConfig_StartRecord(adc_path, 1)

    -- Give DCA1000 time to enter recording mode.
    RSTD.Sleep(pre_trigger_ms)

    WriteToLog(string.format("Scan %d/%d: trigger frame\n", i, num_scans), "green")
    ar1.StartFrame()

    -- Wait for 64-frame capture plus DCA1000 file flush.
    RSTD.Sleep(post_trigger_ms)

    WriteToLog(string.format("Scan %d/%d: capture cycle done\n", i, num_scans), "green")
end

WriteToLog("closed_loop_capture_64f_safe.lua finished\n", "green")