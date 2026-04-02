import subprocess
import os
import time

# -- Test Configuration --
UE_EXE = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
PROJECT_DIR = r"C:/Users/Bia/Documents/Unreal Projects/SubwayTrain/SubwayTrain/SubwayTrain.uproject"
RUNTIME_DIR = os.path.join(os.getcwd(), "test_runtimes")
CSV_FRAMES = 450

# CLEANED PATHS: Format must be /Game/Folder/AssetName.AssetName
test_scenes = {
    "subway-lookdown": {
        "level": "/Game/SubwayTrain/Maps/Demonstration",
        "sequence": "/Game/SubwayTrain/Flythrough-Close.Flythrough-Close"
    },
    "subway-turn": {
        "level": "/Game/SubwayTrain/Maps/Demonstration",
        "sequence": "/Game/SubwayTrain/Flythrough.Flythrough"
    }
}

for name, info in test_scenes.items():
    print(f"\n>>> TESTING SCENE: {name}")
    
    # CHANGE 1: Use 'ce RemoteStartSequence' to talk to your Blueprint
    # CHANGE 2: Ensure it is at the END of the list so CSV and Sleep happen first
    exec_cmds = ",".join([
        "r.FixedFrameRate 30",
        "t.MaxFPS 0",
        "r.VSync 0",
        "r.gpuCsvStatsEnabled 1",
        "Sleep 5", 
        f"csvprofile frames={CSV_FRAMES}",
        "csvprofile start",
        f"ce RemoteStartSequence {info['sequence']}"
    ])

    cmd = [
        UE_EXE, PROJECT_DIR, info['level'],
        "-game", "-windowed", "-ResX=1920", "-ResY=1080",
        "-nosplash", "-novsync", "-log",
        "-Benching",            
        "-NoTextureStreaming",  
        "-NoSound",             # Added to reduce CPU noise
        "-NoVerifyGC",          # Added to prevent hitches during benchmark
        f"-csvdir={RUNTIME_DIR}",
        f"-ExecCmds={exec_cmds}",
        "-ExitAfterCsvProfiling",
    ]

    subprocess.run(cmd)
    print(f">>> Finished {name}. Waiting for engine cleanup...")
    time.sleep(2)