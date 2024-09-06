COMMAND_RESET = {
    "type": "reset",
    "use_agent_ID": [1,2,3,4],
    "value": {
        "done": False,
        "first": True
    },
    "step": 0,
    "episode": 1
}

COMMAND_ACTION = {
    "type": "action",
    "use_agent_ID": [1,2,3,4],
    "value": {
        "offloading_rate": 0.5,
        "resolution": "600x360",
        "bitrate": 3000000,
        "edge_model": "s",
        "cloud_model": "m"
    },
    "step": 0,
    "episode": 1,
    "flag": False
}

COMMAND_KILL = {
    "type": "kill",
    "use_agent_ID": [1,2,3,4],
    "value": {
        "done": False,
        "first": True
    }
}
