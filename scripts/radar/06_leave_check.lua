WriteToLog("leave_check_only.lua started\n", "yellow")
WriteToLog("This script does NOT power off, reset, disconnect, or stop frame.\n", "yellow")
WriteToLog("Please check Output log manually:\n", "yellow")
WriteToLog("  1. Frame Ended\n", "yellow")
WriteToLog("  2. Frame End async event received\n", "yellow")
WriteToLog("  3. Record is completed / Record stop is done successfully\n", "yellow")

ar1.GetMSSFwVersion()
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()

WriteToLog("If no recording/framing is active, close mmWave Studio and then remove board power.\n", "green")
WriteToLog("leave_check_only.lua finished\n", "green")