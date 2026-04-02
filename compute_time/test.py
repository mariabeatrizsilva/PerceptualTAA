import subprocess
import os

# --- Quick Test Config ---
UE_EXE = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
PROJECT = r"C:/Users/Bia/Documents/Unreal Projects/SubwayTrain/SubwayTrain/SubwayTrain.uproject"
LEVEL   = "/Game/SubwayTrain/Maps/Demonstration"

# The two sequences you want to test
TEST_SCENES = [
    {"name": "subway-lookdown", "seq": "/Game/SubwayTrain/Flythrough-Close"},
    {"name": "subway-turn",     "seq": "/Game/SubwayTrain/Flythrough"}
]

for scene in TEST_SCENES:
    print(f"\n🎬 TESTING SEQUENCE: {scene['name']}")
    
    exec_cmds = ",".join([
        "r.FixedFrameRate 30",
        "t.MaxFPS 0",
        "r.gpuCsvStatsEnabled 1",
        "Wait 60",                        # Warm up for 60 frames
        "csvprofile start frames=100",    # Short capture for testing
    ])

    cmd = [
        UE_EXE, PROJECT, LEVEL,
        "-game", "-windowed", "-ResX=1280", "-ResY=720",
        "-nosplash", "-log",
        "-Benching",
        f"-LevelSequence={scene['seq']}", # <--- THIS IS THE MAGIC KEY
        f"-ExecCmds={exec_cmds}",
        "-ExitAfterCsvProfiling",
    ]

    subprocess.run(cmd)

print("\n✅ Test sequence run complete.")