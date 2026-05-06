WriteToLog("hello_safe.lua started\n", "green")
WriteToLog("This script does not touch radar or DCA1000.\n", "green")

-- Optional: query versions only. These are read-only style API calls.
ar1.GetMSSFwVersion()
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()

WriteToLog("hello_safe.lua finished\n", "green")