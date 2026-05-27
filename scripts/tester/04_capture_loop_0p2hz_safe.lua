WriteToLog("capture_loop_0p1hz_safe.lua started\n", "green")

base_dir = "C:\\Users\\user\\Desktop\\ADPAR-final-project\\adc\\"
num_scans = 5

for i = 1, num_scans do
    adc_path = base_dir .. string.format("safe10s_%03d.bin", i)

    WriteToLog(string.format("Scan %d/%d: start record -> %s\n", i, num_scans, adc_path), "green")

    ar1.CaptureCardConfig_StartRecord(adc_path, 1)

    -- Give DCA1000 enough time to enter recording state.
    RSTD.Sleep(3500)

    WriteToLog(string.format("Scan %d/%d: trigger frame\n", i, num_scans), "green")
    ar1.StartFrame()

    -- Your frame is short, but DCA1000 needs time to stop and flush the file.
    RSTD.Sleep(6000)

    WriteToLog(string.format("Scan %d/%d: done, waiting next cycle\n", i, num_scans), "green")
end

WriteToLog("capture_loop_0p1hz_safe.lua finished\n", "green")