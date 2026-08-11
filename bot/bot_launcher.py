#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.mc_bot import MinecraftAIBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('minecraft_ai_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MinecraftAIBotLauncher')


def main():
    parser = argparse.ArgumentParser(description='Minecraft AI Bot - Autonomous Minecraft Player')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Minecraft server host (default: localhost)')
    parser.add_argument('--port', type=int, default=25575,
                       help='AI mod TCP port (default: 25575)')
    parser.add_argument('--fps', type=int, default=10,
                       help='AI decision frequency (default: 10)')
    parser.add_argument('--policy', type=str, default=None,
                       help='Path to pre-trained policy model')
    parser.add_argument('--llm', action='store_true',
                       help='Enable LLM-based decision making')
    parser.add_argument('--goal', type=str, default=None,
                       help='Override initial goal (e.g., GATHER_WOOD)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level (default: INFO)')
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    config = {
        'fps': args.fps,
        'decision': {
            'interval_frames': 50,
            'simulate_on_failure': True,
            'game_type': 'minecraft',
        }
    }

    bot = MinecraftAIBot(
        host=args.host,
        port=args.port,
        use_llm=args.llm,
        policy_path=args.policy,
        config=config,
    )

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("=" * 60)
    logger.info("  Minecraft AI Bot - Autonomous Player System")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  FPS: {args.fps} | LLM: {args.llm}")
    logger.info(f"  Policy: {args.policy or 'default (untrained)'}")
    logger.info("=" * 60)

    if not bot.start():
        logger.error("Failed to start bot. Make sure the Minecraft server with AI mod is running.")
        sys.exit(1)

    if args.goal:
        bot.controller.set_goal(args.goal)
        bot.goal_planner.current_goal = args.goal

    try:
        while bot._running:
            status = bot.get_status()
            if bot._frame_count % 100 == 0:
                logger.info(
                    f"Frame: {status['frames']} | HP: {status['health']:.0f} | "
                    f"Goal: {status['goal']} | {status['goal_planner_status']}"
                )
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()


if __name__ == '__main__':
    main()
