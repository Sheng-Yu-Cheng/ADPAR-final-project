WriteToLog("closed_loop_capture_safe.lua started\n", "green")

base_dir = "C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\"
decision_path = base_dir .. "latest_decision.txt"

num_scans = 10

-- Start conservative. Later we can reduce these.
pre_trigger_ms = 3500
post_trigger_ms = 6000

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

    -- Safe stop condition. Python can write STOP as first line of latest_decision.txt.
    if prev_decision == "STOP" then
        WriteToLog("STOP decision received. Exiting loop safely.\n", "red")
        break
    end

    local adc_path = base_dir .. string.format("cl_%03d.bin", i)

    WriteToLog(string.format("Scan %d/%d: start record -> %s\n", i, num_scans, adc_path), "green")

    ar1.CaptureCardConfig_StartRecord(adc_path, 1)

    -- Give DCA1000 time to enter record state.
    RSTD.Sleep(pre_trigger_ms)

    WriteToLog(string.format("Scan %d/%d: trigger frame\n", i, num_scans), "green")
    ar1.StartFrame()

    -- Give DCA1000 time to finish and flush file.
    RSTD.Sleep(post_trigger_ms)

    WriteToLog(string.format("Scan %d/%d: capture cycle done\n", i, num_scans), "green")
end

WriteToLog("closed_loop_capture_safe.lua finished\n", "green")