import time

from controls.episode_manager import EpisodeConfig, EpisodeManager
from controls.setup import setup_components, shutdown_components
from gym_env.ai_gp_env import AIGPEnv
from scripts.reset_hotkey import ResetHotkey


def create_official_env(
    *,
    sim_ip="127.0.0.1",
    sim_port=14550,
    vision_ip="0.0.0.0",
    vision_port=5600,
    episode_config=None,
):
    """Connect one Gym environment to one official Windows simulator instance."""
    data = {}
    components = setup_components(
        data,
        int(time.time() * 1000),
        sim_ip,
        sim_port,
        vision_ip,
        vision_port,
    )
    controller = components["controller"]
    episodes = EpisodeManager(
        data,
        send_reset=controller.send_sim_reset_command,
        send_arm=controller.arm,
        config=episode_config or EpisodeConfig(),
    )
    hotkey = ResetHotkey()
    return AIGPEnv(
        data,
        controller,
        episodes,
        manual_reset_requested=hotkey.consume_request,
        close_callbacks=(
            hotkey.close,
            lambda: shutdown_components(components),
        ),
    )
