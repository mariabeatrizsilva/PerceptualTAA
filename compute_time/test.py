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
    
    play_seq_cmd = f"LevelSequence.Play {scene['seq']}"

    exec_cmds = ",".join([
        "r.FixedFrameRate 30",
        "t.MaxFPS 0",
        play_seq_cmd,                 # <--- Force play via console command
        "Wait 60",
        "csvprofile start frames=100",
    ])

    cmd = [
        UE_EXE, PROJECT, LEVEL,
        "-game", "-windowed", "-ResX=1280", "-ResY=720",
        "-nosplash", "-log",
        "-Benching",
        f"-ExecCmds={exec_cmds}",     # Use this instead of the -LevelSequence flag
        "-ExitAfterCsvProfiling",
    ]
    subprocess.run(cmd)

print("\n✅ Test sequence run complete.")