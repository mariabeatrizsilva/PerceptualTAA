import subprocess
import os
import time

# -- Test Configuration --
#### -- before do this turn off blueprint for both!!
UE_EXE = r"C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
# PROJECT_DIR = r"C:\Users\YourUser\Documents\Unreal Projects\SubwayTrain\SubwayTrain.uproject"
PROJECT_DIR = r"C:/Users/Bia/Documents/Unreal Projects/SubwayTrain/SubwayTrain/SubwayTrain.uproject"
RUNTIME_DIR = os.path.join(os.getcwd(), "test_runtimes")
CSV_FRAMES = 450

test_scenes = {
    "subway-lookdown": {
        "level": "/Game/SubwayTrain/Maps/Demonstration",
        "sequence": "/Game/SubwayTrain/Flythrough-Close.Flythrough"
    },
    "subway-turn": {
        "level": "/Game/SubwayTrain/Maps/Demonstration",
        "sequence": "/Script/LevelSequence.LevelSequence'/Game/SubwayTrain/Flythrough.Flythrough'"
    }
}

# /Script/LevelSequence.LevelSequence'/Game/SubwayTrain/Flythrough.Flythrough'
for name, info in test_scenes.items():
    print(f"\n>>> TESTING SCENE: {name}")
    
    # We use 'Play Looping' so it covers the full 450 frames of our CSV
    exec_cmds = ",".join([
        "r.FixedFrameRate 30",
        "t.MaxFPS 0",
        "r.VSync 0",
        "r.gpuCsvStatsEnabled 1",
        "Sleep 5", # Warm up shaders/textures
        f"csvprofile frames={CSV_FRAMES}",
        "csvprofile start",
        f"LevelSequencePlayer.CreateLevelSequencePlayer {info['sequence']} Play",
    ])

    cmd = [
        UE_EXE, PROJECT_DIR, info['level'],
        "-game", "-windowed", "-ResX=1920", "-ResY=1080",
        "-nosplash", "-novsync", "-log",
        "-Benching",            # Keep timings deterministic
        "-NoTextureStreaming",  # Reduce IO noise
        f"-csvdir={RUNTIME_DIR}",
        f"-ExecCmds={exec_cmds}",
        "-ExitAfterCsvProfiling",
    ]

    subprocess.run(cmd)
    print(f">>> Finished {name}. Waiting for engine cleanup...")
    time.sleep(2)