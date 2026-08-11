package com.gameaisystem.minecraftmod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.entity.FakePlayer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.GameMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MinecraftAIMod implements ModInitializer {
    public static final String MOD_ID = "minecraftaimod";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    public static AIPlayer aiPlayer;
    public static AITCPServer tcpServer;
    public static final ConcurrentLinkedQueue<String> commandQueue = new ConcurrentLinkedQueue<>();
    public static final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

    @Override
    public void onInitialize() {
        LOGGER.info("Minecraft AI Mod initializing...");

        ServerLifecycleEvents.SERVER_STARTED.register(server -> {
            LOGGER.info("Server started, setting up AI components...");
            new Thread(() -> {
                try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
                tcpServer = new AITCPServer(25575);
                tcpServer.start();
                LOGGER.info("AI TCP Server started on port 25575");
            }, "AI-TCP-Server-Init").start();
        });

        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            LOGGER.info("Server stopping, cleaning up AI...");
            if (tcpServer != null) tcpServer.stop();
            if (aiPlayer != null) aiPlayer.cleanup();
        });

        ServerTickEvents.END_SERVER_TICK.register(server -> {
            if (aiPlayer != null && aiPlayer.isActive()) {
                aiPlayer.tick();
            }
            String cmd;
            while ((cmd = commandQueue.poll()) != null) {
                processCommand(server, cmd);
            }
        });

        registerCommands();
        LOGGER.info("Minecraft AI Mod initialized successfully");
    }

    private void registerCommands() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
            dispatcher.register(
                net.minecraft.server.command.CommandManager.literal("aiplayer")
                    .then(net.minecraft.server.command.CommandManager.literal("spawn")
                        .executes(ctx -> {
                            ServerPlayerEntity source = ctx.getSource().getPlayer();
                            if (source == null) return 0;
                            BlockPos spawnPos = source.getBlockPos();
                            spawnAIPlayer(ctx.getSource().getServer(), ctx.getSource().getWorld(), spawnPos);
                            ctx.getSource().sendFeedback(() -> Text.literal(
                                "AI Player spawned at " + spawnPos.getX() + ", " + spawnPos.getY() + ", " + spawnPos.getZ()
                            ), true);
                            return 1;
                        })
                    )
                    .then(net.minecraft.server.command.CommandManager.literal("start")
                        .executes(ctx -> {
                            if (aiPlayer != null) {
                                aiPlayer.setActive(true);
                                ctx.getSource().sendFeedback(() -> Text.literal("AI Player activated"), true);
                            } else {
                                ctx.getSource().sendFeedback(() -> Text.literal("No AI Player spawned. Use /aiplayer spawn first."), false);
                            }
                            return 1;
                        })
                    )
                    .then(net.minecraft.server.command.CommandManager.literal("stop")
                        .executes(ctx -> {
                            if (aiPlayer != null) {
                                aiPlayer.setActive(false);
                                ctx.getSource().sendFeedback(() -> Text.literal("AI Player deactivated"), true);
                            }
                            return 1;
                        })
                    )
                    .then(net.minecraft.server.command.CommandManager.literal("status")
                        .executes(ctx -> {
                            if (aiPlayer == null) {
                                ctx.getSource().sendFeedback(() -> Text.literal("AI Player: not spawned"), false);
                            } else {
                                ctx.getSource().sendFeedback(() -> Text.literal(
                                    "AI Player: " + (aiPlayer.isActive() ? "ACTIVE" : "PAUSED") +
                                    " | Goal: " + aiPlayer.getCurrentGoal() +
                                    " | Pos: " + aiPlayer.getPlayer().getBlockPos().toShortString()
                                ), false);
                            }
                            return 1;
                        })
                    )
                    .then(net.minecraft.server.command.CommandManager.literal("teleport")
                        .executes(ctx -> {
                            ServerPlayerEntity source = ctx.getSource().getPlayer();
                            if (source != null && aiPlayer != null && aiPlayer.isActive()) {
                                aiPlayer.teleportTo(source.getBlockPos());
                                ctx.getSource().sendFeedback(() -> Text.literal("AI Player teleported to your position"), true);
                            }
                            return 1;
                        })
                    )
            );
        });
    }

    public void spawnAIPlayer(MinecraftServer server, ServerWorld world, BlockPos pos) {
        if (aiPlayer != null) {
            aiPlayer.cleanup();
        }
        aiPlayer = new AIPlayer(server, world, pos);
        aiPlayer.spawn();
        LOGGER.info("AI Player spawned at {} in dimension {}", pos, world.getRegistryKey().getValue());
    }

    private void processCommand(MinecraftServer server, String raw) {
        try {
            String[] parts = raw.split(" ", 3);
            String type = parts[0];
            switch (type) {
                case "SPAWN":
                    if (aiPlayer != null) aiPlayer.cleanup();
                    ServerWorld world = server.getWorld(net.minecraft.registry.RegistryKey.of(
                        net.minecraft.registry.RegistryKeys.WORLD,
                        new net.minecraft.util.Identifier(parts.length > 1 ? parts[1] : "minecraft:overworld")
                    ));
                    if (world != null) {
                        BlockPos pos = world.getSpawnPos();
                        spawnAIPlayer(server, world, pos);
                    }
                    break;
                case "START":
                    if (aiPlayer != null) aiPlayer.setActive(true);
                    break;
                case "STOP":
                    if (aiPlayer != null) aiPlayer.setActive(false);
                    break;
                case "TELEPORT":
                    if (aiPlayer != null && parts.length > 2) {
                        String[] coords = parts[2].split(",");
                        BlockPos tpPos = new BlockPos(
                            Integer.parseInt(coords[0]),
                            Integer.parseInt(coords[1]),
                            Integer.parseInt(coords[2])
                        );
                        aiPlayer.teleportTo(tpPos);
                    }
                    break;
                case "SET_GOAL":
                    if (aiPlayer != null && parts.length > 1) {
                        aiPlayer.setGoal(parts[1]);
                    }
                    break;
                case "GET_WORLD_STATE":
                    if (aiPlayer != null && tcpServer != null) {
                        String state = aiPlayer.getWorldStateJSON();
                        tcpServer.sendToAll(state);
                    }
                    break;
                case "SHUTDOWN":
                    if (aiPlayer != null) aiPlayer.cleanup();
                    if (tcpServer != null) tcpServer.stop();
                    break;
                default:
                    if (aiPlayer != null && aiPlayer.isActive()) {
                        aiPlayer.executeDirectAction(raw);
                    }
            }
        } catch (Exception e) {
            LOGGER.error("Failed to process command: {}", raw, e);
        }
    }
}
