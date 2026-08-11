import json
import socket
import threading
import time
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

from src.minecraft.mc_perception import MCWorldState

logger = logging.getLogger(__name__)


@dataclass
class BotCommand:
    action: str
    args: List[str]


class MCController:
    def __init__(self, host: str = 'localhost', port: int = 25575):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self._receive_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._state_listeners: List[Callable] = []
        self._latest_state: Optional[MCWorldState] = None
        self._read_buffer = b""

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)
            self.connected = True
            self._running = True

            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            logger.info(f"Connected to Minecraft AI mod at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Minecraft AI mod: {e}")
            return False

    def disconnect(self):
        self._running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        logger.info("Disconnected from Minecraft AI mod")

    def send_command(self, command: str) -> bool:
        if not self.connected or not self.socket:
            return False
        try:
            with self._lock:
                data = (command + '\n').encode('utf-8')
                self.socket.sendall(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self.connected = False
            return False

    def spawn_ai_player(self) -> bool:
        return self.send_command("SPAWN")

    def start_ai(self) -> bool:
        return self.send_command("START")

    def stop_ai(self) -> bool:
        return self.send_command("STOP")

    def set_goal(self, goal: str) -> bool:
        return self.send_command(f"SET_GOAL {goal}")

    def request_state(self) -> bool:
        return self.send_command("GET_WORLD_STATE")

    def execute_action(self, action: str) -> bool:
        return self.send_command(action)

    def get_latest_state(self) -> Optional[MCWorldState]:
        if self._latest_state is None:
            self.request_state()
            time.sleep(0.05)
        return self._latest_state

    def add_state_listener(self, callback: Callable):
        self._state_listeners.append(callback)

    def remove_state_listener(self, callback: Callable):
        if callback in self._state_listeners:
            self._state_listeners.remove(callback)

    def _receive_loop(self):
        while self._running and self.connected:
            try:
                if self.socket is None:
                    break
                data = self.socket.recv(4096)
                if not data:
                    logger.warning("Connection closed by server")
                    self.connected = False
                    break

                self._read_buffer += data
                while b'\n' in self._read_buffer:
                    line, self._read_buffer = self._read_buffer.split(b'\n', 1)
                    self._process_message(line.decode('utf-8').strip())

            except socket.timeout:
                continue
            except ConnectionResetError:
                logger.warning("Connection reset by server")
                self.connected = False
                break
            except Exception as e:
                if self._running:
                    logger.error(f"Receive error: {e}")
                break

        self._running = False

    def _process_message(self, message: str):
        if not message:
            return
        try:
            if message.startswith('{'):
                state = MCWorldState.from_json(message)
                self._latest_state = state
                for listener in self._state_listeners:
                    try:
                        listener(state)
                    except Exception as e:
                        logger.error(f"State listener error: {e}")
            elif '"type":"connected"' in message:
                logger.info("Successfully registered with Minecraft AI mod")
            else:
                logger.debug(f"Mod message: {message}")
        except Exception as e:
            logger.debug(f"Non-JSON message: {message[:100]}")

    def is_connected(self) -> bool:
        return self.connected


class MCCommandTranslator:
    def __init__(self):
        self.block_map = {
            'oak_log': 'minecraft:oak_log',
            'stone': 'minecraft:stone',
            'cobblestone': 'minecraft:cobblestone',
            'dirt': 'minecraft:dirt',
            'iron_ore': 'minecraft:iron_ore',
            'diamond_ore': 'minecraft:diamond_ore',
            'coal_ore': 'minecraft:coal_ore',
            'gold_ore': 'minecraft:gold_ore',
        }

    def break_block(self, x: int, y: int, z: int) -> str:
        return f"BREAK:{x},{y},{z}"

    def place_block(self, x: int, y: int, z: int) -> str:
        return f"PLACE:{x},{y},{z}"

    def use_block(self, x: int, y: int, z: int) -> str:
        return f"USE:{x},{y},{z}"

    def attack_entity(self) -> str:
        return "ATTACK"

    def equip_slot(self, slot: int) -> str:
        return f"EQUIP:{slot}"

    def drop_item(self) -> str:
        return "DROP"

    def craft(self, item: str) -> str:
        return f"CRAFT:{item}"

    def eat(self) -> str:
        return "EAT"

    def sleep(self) -> str:
        return "SLEEP"

    def move_to(self, x: float, y: float, z: float) -> str:
        return f"MOVE:{x:.1f}:{y:.1f}:{z:.1f}"

    def look_at(self, yaw: float, pitch: float) -> str:
        return f"LOOK:{yaw:.1f}:{pitch:.1f}"

    def start_mining(self) -> str:
        return "MINE"

    def start_exploring(self) -> str:
        return "EXPLORE"
