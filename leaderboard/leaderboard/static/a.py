import json

data = {
    "ResNet50": [
        {"kernel_name": "Kernel A", "leaderboard_id": 1},
        {"kernel_name": "Kernel B", "leaderboard_id": 2}
    ],
    "Conv2D": [
        {"kernel_name": "Conv Layer 1", "leaderboard_id": 3}
    ],
    "Model 3": [
        {"kernel_name": "Kernel A", "leaderboard_id": 2},
        {"kernel_name": "Kernel B", "leaderboard_id": 2}
    ]
}

with open("framework_kernels.jsonl", "w") as f:
    for framework, kernels in data.items():
        for kernel in kernels:
            entry = {
                "framework": framework,
                "kernel_name": kernel["kernel_name"],
                "leaderboard_id": kernel["leaderboard_id"]
            }
            f.write(json.dumps(entry) + "\n")

print("✅ JSONL file written!")
