package com.gameaisystem.minecraftmod;

import com.mojang.authlib.GameProfile;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.attribute.EntityAttributes;
import net.minecraft.entity.damage.DamageSource;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.passive.AnimalEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.*;
import net.minecraft.recipe.Recipe;
import net.minecraft.recipe.RecipeManager;
import net.minecraft.registry.Registries;
import net.minecraft.screen.CraftingScreenHandler;
import net.minecraft.screen.PlayerScreenHandler;
import net.minecraft.screen.ScreenHandler;
import net.minecraft.screen.slot.SlotActionType;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayNetworkHandler;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ChunkTicketType;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.Identifier;
import net.minecraft.util.Unit;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.*;
import net.minecraft.world.GameMode;
import net.minecraft.world.RaycastContext;
import net.minecraft.world.World;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class AIPlayer {
    private static final Logger LOGGER = LoggerFactory.getLogger("AIPlayer");
    private static final int CHUNK_LOAD_RADIUS = 4;
    private static final int STATE_UPDATE_INTERVAL = 20;
    private static final double REACH_DISTANCE = 5.0;
    private static final UUID AI_UUID = UUID.fromString("c0d3b33f-a1b0-4f10-91a1-b07c0de30001");

    private final MinecraftServer server;
    private final ServerWorld world;
    private ServerPlayerEntity player;
    private boolean active = false;
    private int tickCounter = 0;
    private int stateUpdateTimer = 0;
    private ChunkPos lastChunkPos;
    private final List<ChunkPos> loadedChunks = new ArrayList<>();

    private Goal currentGoal = Goal.GATHER_WOOD;
    private final GoalPlanner goalPlanner;
    private final Pathfinder pathfinder;
    private final CombatController combatController;
    private final InventoryManager inventoryManager;

    private Vec3d targetPosition;
    private String lastWorldState;
    private int idleTicks = 0;
    private boolean hasRegistered = false;

    public AIPlayer(MinecraftServer server, ServerWorld world, BlockPos spawnPos) {
        this.server = server;
        this.world = world;
        this.goalPlanner = new GoalPlanner(this);
        this.pathfinder = new Pathfinder();
        this.combatController = new CombatController(this);
        this.inventoryManager = new InventoryManager(this);
        this.targetPosition = Vec3d.ofCenter(spawnPos);
    }

    public void spawn() {
        GameProfile profile = new GameProfile(AI_UUID, "AIPlayer");
        this.player = new ServerPlayerEntity(server, world, profile);
        this.player.changeGameMode(GameMode.SURVIVAL);
        this.player.refreshPositionAndAngles(targetPosition.x, targetPosition.y, targetPosition.z, 0, 0);
        this.player.networkHandler = new ServerPlayNetworkHandler(server, new net.minecraft.network.ClientConnection(net.minecraft.network.NetworkSide.SERVERBOUND), player);
        world.spawnEntity(player);
        server.getPlayerManager().onPlayerConnect(player.networkHandler);
        this.player.setHealth(this.player.getMaxHealth());
        this.player.getHungerManager().setFoodLevel(20);
        this.active = true;
        this.hasRegistered = true;
        loadChunksAround();
        LOGGER.info("AI Player spawned and connected");
    }

    public void cleanup() {
        active = false;
        if (player != null && hasRegistered) {
            try {
                unloadAllChunks();
                server.getPlayerManager().remove(player);
            } catch (Exception e) {
                LOGGER.warn("Error cleaning up AI player: {}", e.getMessage());
            }
        }
        player = null;
        hasRegistered = false;
    }

    public boolean isActive() { return active && player != null; }
    public void setActive(boolean active) { this.active = active; }
    public ServerPlayerEntity getPlayer() { return player; }
    public ServerWorld getWorld() { return world; }
    public String getCurrentGoal() { return currentGoal.name(); }

    public void setGoal(String goalName) {
        try {
            this.currentGoal = Goal.valueOf(goalName.toUpperCase());
            goalPlanner.reset();
            LOGGER.info("AI Player goal set to {}", currentGoal);
        } catch (IllegalArgumentException e) {
            LOGGER.warn("Unknown goal: {}", goalName);
        }
    }

    public void teleportTo(BlockPos pos) {
        if (player != null) {
            player.teleport(world, pos.getX() + 0.5, pos.getY(), pos.getZ() + 0.5,
                player.getYaw(), player.getPitch());
            targetPosition = Vec3d.ofCenter(pos);
            loadChunksAround();
        }
    }

    public void tick() {
        if (!isActive()) return;

        tickCounter++;
        stateUpdateTimer++;

        loadChunksAround();

        combatController.checkThreats();
        if (combatController.isInCombat()) {
            combatController.tick();
        } else {
            goalPlanner.tick();
            if (goalPlanner.needsMovement()) {
                moveTowardTarget();
            }
        }

        if (tickCounter % 40 == 0) {
            inventoryManager.organize();
        }

        if (stateUpdateTimer >= STATE_UPDATE_INTERVAL) {
            stateUpdateTimer = 0;
            updateWorldState();
        }
    }

    private void loadChunksAround() {
        if (player == null) return;
        ChunkPos currentPos = player.getChunkPos();
        if (currentPos.equals(lastChunkPos)) return;
        lastChunkPos = currentPos;

        unloadAllChunks();

        for (int dx = -CHUNK_LOAD_RADIUS; dx <= CHUNK_LOAD_RADIUS; dx++) {
            for (int dz = -CHUNK_LOAD_RADIUS; dz <= CHUNK_LOAD_RADIUS; dz++) {
                ChunkPos chunkPos = new ChunkPos(currentPos.x + dx, currentPos.z + dz);
                Vec3d center = new Vec3d(
                    chunkPos.getCenterX(), world.getBottomY(), chunkPos.getCenterZ()
                );
                world.getChunkManager().addTicket(
                    ChunkTicketType.create("ai_player", Comparator.comparingLong(c -> {
                        ChunkPos cp = (ChunkPos) c;
                        return cp.toLong();
                    }), 1200),
                    chunkPos, 3, Unit.INSTANCE
                );
                loadedChunks.add(chunkPos);
            }
        }
    }

    private void unloadAllChunks() {
        for (ChunkPos chunkPos : loadedChunks) {
            try {
                world.getChunkManager().removeTicket(
                    ChunkTicketType.create("ai_player", Comparator.comparingLong(c -> {
                        ChunkPos cp = (ChunkPos) c;
                        return cp.toLong();
                    }), 1200),
                    chunkPos, 3, Unit.INSTANCE
                );
            } catch (Exception ignored) {}
        }
        loadedChunks.clear();
    }

    private void moveTowardTarget() {
        if (player == null || targetPosition == null) return;

        Vec3d current = player.getPos();
        double dx = targetPosition.x - current.x;
        double dy = targetPosition.y - current.y;
        double dz = targetPosition.z - current.z;
        double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < 0.5) {
            idleTicks++;
            return;
        }

        idleTicks = 0;
        Vec3d direction = new Vec3d(dx, dy, dz).normalize();
        float yaw = (float) (Math.atan2(dz, dx) * 180.0 / Math.PI) - 90.0f;
        float pitch = (float) (-Math.atan2(dy, Math.sqrt(dx * dx + dz * dz)) * 180.0 / Math.PI);

        player.setYaw(yaw);
        player.setPitch(pitch);
        player.setHeadYaw(yaw);

        player.setVelocity(direction.multiply(0.3));
        player.setSprinting(dist > 5.0);

        if (dist < 1.5 && Math.abs(dy) < 1.0) {
            player.jump();
        }
    }

    public boolean breakBlock(BlockPos blockPos) {
        if (player == null) return false;
        if (blockPos.getSquaredDistance(player.getPos()) > REACH_DISTANCE * REACH_DISTANCE) return false;

        BlockState blockState = world.getBlockState(blockPos);
        if (blockState.isAir()) return true;

        player.interactionManager.tryBreakBlock(blockPos);
        return world.getBlockState(blockPos).isAir();
    }

    public boolean placeBlock(BlockPos blockPos, ItemStack item) {
        if (player == null || item == null || !(item.getItem() instanceof BlockItem)) return false;
        if (blockPos.getSquaredDistance(player.getPos()) > REACH_DISTANCE * REACH_DISTANCE) return false;

        BlockItem blockItem = (BlockItem) item.getItem();
        BlockHitResult hit = new BlockHitResult(Vec3d.ofCenter(blockPos), Direction.UP, blockPos, false);
        ActionResult result = blockItem.place(new ItemPlacementContext(player, Hand.MAIN_HAND, item, hit));
        if (result == ActionResult.SUCCESS) {
            item.decrement(1);
            return true;
        }
        return false;
    }

    public ItemStack findItem(String itemName) {
        if (player == null) return ItemStack.EMPTY;
        PlayerInventory inv = player.getInventory();
        for (int i = 0; i < inv.size(); i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty()) {
                Identifier id = Registries.ITEM.getId(stack.getItem());
                if (id.toString().contains(itemName.toLowerCase()) ||
                    id.getPath().contains(itemName.toLowerCase())) {
                    return stack;
                }
            }
        }
        return ItemStack.EMPTY;
    }

    public boolean hasItem(String itemName, int count) {
        if (player == null) return false;
        int total = 0;
        PlayerInventory inv = player.getInventory();
        for (int i = 0; i < inv.size(); i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty()) {
                Identifier id = Registries.ITEM.getId(stack.getItem());
                if (id.toString().contains(itemName.toLowerCase()) ||
                    id.getPath().contains(itemName.toLowerCase())) {
                    total += stack.getCount();
                }
            }
        }
        return total >= count;
    }

    public boolean selectBestTool(BlockState blockState) {
        if (player == null) return false;
        PlayerInventory inv = player.getInventory();
        int bestSlot = inv.selectedSlot;
        float bestSpeed = 1.0f;

        for (int i = 0; i < 9; i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty()) {
                float speed = stack.getMiningSpeedMultiplier(blockState);
                if (speed > bestSpeed) {
                    bestSpeed = speed;
                    bestSlot = i;
                }
            }
        }
        inv.selectedSlot = bestSlot;
        return true;
    }

    public List<HostileEntity> getNearbyHostiles() {
        List<HostileEntity> hostiles = new ArrayList<>();
        if (player == null) return hostiles;

        Box box = player.getBoundingBox().expand(16);
        List<LivingEntity> entities = world.getEntitiesByClass(LivingEntity.class, box,
            e -> e instanceof HostileEntity && e.isAlive());
        for (LivingEntity e : entities) {
            hostiles.add((HostileEntity) e);
        }
        return hostiles;
    }

    public List<ItemEntity> getNearbyItems(double radius) {
        List<ItemEntity> items = new ArrayList<>();
        if (player == null) return items;

        Box box = player.getBoundingBox().expand(radius);
        List<ItemEntity> entities = world.getEntitiesByClass(ItemEntity.class, box, e -> true);
        return entities;
    }

    public BlockPos findNearbyBlock(String blockId, int radius) {
        BlockPos.Mutable mbp = new BlockPos.Mutable();
        BlockPos origin = player.getBlockPos();
        for (int y = -8; y <= 8; y++) {
            for (int x = -radius; x <= radius; x++) {
                for (int z = -radius; z <= radius; z++) {
                    mbp.set(origin.getX() + x, origin.getY() + y, origin.getZ() + z);
                    Block block = world.getBlockState(mbp).getBlock();
                    if (Registries.BLOCK.getId(block).toString().contains(blockId)) {
                        return mbp.toImmutable();
                    }
                }
            }
        }
        return null;
    }

    public void equipBestArmor() {
        if (player == null) return;
        PlayerInventory inv = player.getInventory();
        EquipmentSlot[] slots = {EquipmentSlot.HEAD, EquipmentSlot.CHEST,
            EquipmentSlot.LEGS, EquipmentSlot.FEET};

        for (EquipmentSlot slot : slots) {
            ItemStack current = player.getEquippedStack(slot);
            int bestArmor = current.isEmpty() ? 0 : getArmorValue(current);

            for (int i = 9; i < inv.size(); i++) {
                ItemStack stack = inv.getStack(i);
                if (!stack.isEmpty() && stack.getItem() instanceof ArmorItem armor) {
                    if (armor.getSlotType() == slot) {
                        int armorValue = getArmorValue(stack);
                        if (armorValue > bestArmor) {
                            bestArmor = armorValue;
                            inv.setStack(i, current.copy());
                            player.equipStack(slot, stack);
                        }
                    }
                }
            }
        }
    }

    private int getArmorValue(ItemStack stack) {
        if (stack.isEmpty() || !(stack.getItem() instanceof ArmorItem armor)) return 0;
        return armor.getProtection();
    }

    public String getWorldStateJSON() {
        if (player == null) return createEmptyState();

        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"health\":").append(player.getHealth()).append(",");
        json.append("\"maxHealth\":").append(player.getMaxHealth()).append(",");
        json.append("\"hunger\":").append(player.getHungerManager().getFoodLevel()).append(",");
        json.append("\"saturation\":").append(player.getHungerManager().getSaturationLevel()).append(",");
        json.append("\"posX\":").append(String.format("%.1f", player.getX())).append(",");
        json.append("\"posY\":").append(String.format("%.1f", player.getY())).append(",");
        json.append("\"posZ\":").append(String.format("%.1f", player.getZ())).append(",");
        json.append("\"yaw\":").append(String.format("%.1f", player.getYaw())).append(",");
        json.append("\"pitch\":").append(String.format("%.1f", player.getPitch())).append(",");
        json.append("\"dimension\":\"").append(world.getRegistryKey().getValue()).append("\",");
        json.append("\"biome\":\"").append(world.getBiome(player.getBlockPos()).getKey().orElse(
            Identifier.of("unknown")).toString()).append("\",");
        json.append("\"currentGoal\":\"").append(currentGoal.name()).append("\",");
        json.append("\"inCombat\":").append(combatController.isInCombat()).append(",");
        json.append("\"gamemode\":\"").append(player.interactionManager.getGameMode().name()).append("\",");
        json.append("\"experience\":").append(player.experienceLevel).append(",");
        json.append("\"timeOfDay\":").append(world.getTimeOfDay() % 24000).append(",");
        json.append("\"isRaining\":").append(world.isRaining()).append(",");
        json.append("\"isNight\":").append(world.isNight()).append(",");
        json.append("\"armorProtection\":").append(player.getArmor()).append(",");
        json.append("\"isSprinting\":").append(player.isSprinting()).append(",");
        json.append("\"isSneaking\":").append(player.isSneaking()).append(",");
        json.append("\"isSwimming\":").append(player.isSwimming()).append(",");
        json.append("\"onGround\":").append(player.isOnGround()).append(",");
        json.append("\"isInWater\":").append(player.isTouchingWater()).append(",");
        json.append("\"isInLava\":").append(player.isInLava()).append(",");

        json.append("\"nearbyHostiles\":").append(getNearbyHostiles().size()).append(",");
        json.append("\"nearbyItems\":").append(getNearbyItems(8).size()).append(",");

        json.append("\"inventory\":[");
        PlayerInventory inv = player.getInventory();
        boolean first = true;
        for (int i = 0; i < inv.size(); i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty()) {
                if (!first) json.append(",");
                first = false;
                json.append("{");
                json.append("\"slot\":").append(i).append(",");
                json.append("\"item\":\"").append(Registries.ITEM.getId(stack.getItem()).toString()).append("\",");
                json.append("\"count\":").append(stack.getCount());
                json.append("}");
            }
        }
        json.append("],");

        json.append("\"hotbar\":[");
        first = true;
        for (int i = 0; i < 9; i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty()) {
                if (!first) json.append(",");
                first = false;
                json.append("{");
                json.append("\"slot\":").append(i).append(",");
                json.append("\"item\":\"").append(Registries.ITEM.getId(stack.getItem()).toString()).append("\",");
                json.append("\"count\":").append(stack.getCount());
                json.append("}");
            }
        }
        json.append("],");

        json.append("\"selectedSlot\":").append(inv.selectedSlot).append(",");

        json.append("\"blockLookingAt\":");
        BlockHitResult br = player.raycast(REACH_DISTANCE, 0.0F, false);
        if (br != null && br.getType() == HitResult.Type.BLOCK) {
            json.append("\"").append(Registries.BLOCK.getId(world.getBlockState(br.getBlockPos()).getBlock())).append("\"");
        } else {
            json.append("null");
        }
        json.append("}");

        return json.toString();
    }

    private String createEmptyState() {
        return "{\"health\":0,\"maxHealth\":20,\"hunger\":0,\"posX\":0,\"posY\":0,\"posZ\":0,\"currentGoal\":\"NONE\",\"inCombat\":false,\"nearbyHostiles\":0,\"inventory\":[],\"hotbar\":[],\"blockLookingAt\":null}";
    }

    private void updateWorldState() {
        lastWorldState = getWorldStateJSON();
        if (MinecraftAIMod.tcpServer != null) {
            MinecraftAIMod.tcpServer.sendToAll(lastWorldState);
        }
    }

    public void executeDirectAction(String action) {
        try {
            if (player == null) return;
            String[] parts = action.split(":");
            switch (parts[0]) {
                case "BREAK":
                    if (parts.length > 2) {
                        BlockPos bp = parseBlockPos(parts[1]);
                        breakBlock(bp);
                    }
                    break;
                case "PLACE":
                    if (parts.length > 2) {
                        BlockPos bp = parseBlockPos(parts[1]);
                        placeBlock(bp, player.getMainHandStack());
                    }
                    break;
                case "USE":
                    if (parts.length > 2) {
                        BlockPos bp = parseBlockPos(parts[1]);
                        useBlock(bp);
                    }
                    break;
                case "EQUIP":
                    if (parts.length > 1) {
                        int slot = Integer.parseInt(parts[1]);
                        if (slot >= 0 && slot < 9) {
                            player.getInventory().selectedSlot = slot;
                        }
                    }
                    break;
                case "DROP":
                    player.dropSelectedItem(false);
                    break;
                case "CRAFT":
                    tryCraft(parts.length > 1 ? parts[1] : "crafting_table");
                    break;
                case "SMELT":
                    trySmelt();
                    break;
                case "SLEEP":
                    BlockPos bedPos = findNearbyBlock("bed", 16);
                    if (bedPos != null) {
                        useBlock(bedPos);
                    }
                    break;
                case "EAT":
                    eatFood();
                    break;
                case "MOVE":
                    if (parts.length > 3) {
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        double z = Double.parseDouble(parts[3]);
                        targetPosition = new Vec3d(x, y, z);
                    }
                    break;
                case "LOOK":
                    if (parts.length > 2) {
                        float newYaw = Float.parseFloat(parts[1]);
                        float newPitch = parts.length > 2 ? Float.parseFloat(parts[2]) : 0;
                        player.setYaw(newYaw);
                        player.setHeadYaw(newYaw);
                        player.setPitch(newPitch);
                    }
                    break;
                case "ATTACK":
                    combatController.attackNearest();
                    break;
                case "MINE":
                    goalPlanner.startMining();
                    break;
                case "EXPLORE":
                    targetPosition = player.getPos().add(
                        (Math.random() - 0.5) * 60, 0, (Math.random() - 0.5) * 60
                    );
                    break;
            }
        } catch (Exception e) {
            LOGGER.error("Failed to execute action: {}", action, e);
        }
    }

    private BlockPos parseBlockPos(String coords) {
        String[] parts = coords.split(",");
        return new BlockPos(
            Integer.parseInt(parts[0]),
            Integer.parseInt(parts[1]),
            Integer.parseInt(parts[2])
        );
    }

    private void useBlock(BlockPos pos) {
        if (player == null) return;
        player.interactionManager.interactBlock(player, Hand.MAIN_HAND,
            new BlockHitResult(Vec3d.ofCenter(pos), Direction.UP, pos, false));
    }

    private void tryCraft(String targetItem) {
        if (player == null) return;
        PlayerInventory inv = player.getInventory();

        if (hasItem("log", 1) || hasItem("oak_log", 1) || hasItem("birch_log", 1) ||
            hasItem("spruce_log", 1) || hasItem("jungle_log", 1) || hasItem("acacia_log", 1) ||
            hasItem("dark_oak_log", 1) || hasItem("mangrove_log", 1) || hasItem("cherry_log", 1)) {
            if (!hasItem("planks", 4)) {
                for (int i = 0; i < inv.size(); i++) {
                    ItemStack stack = inv.getStack(i);
                    Identifier id = Registries.ITEM.getId(stack.getItem());
                    if (id.getPath().contains("log")) {
                        inv.selectedSlot = i < 9 ? i : inv.selectedSlot;
                        ItemStack planks = new ItemStack(Registries.ITEM.get(Identifier.of(
                            id.getNamespace(), id.getPath().replace("log", "planks")
                        )), 4);
                        player.getInventory().removeOne(stack);
                        player.getInventory().offerOrDrop(planks);
                        break;
                    }
                }
            }
        }

        if (targetItem.contains("pickaxe") && hasItem("planks", 3) && hasItem("stick", 2)) {
            removeItems("planks", 3);
            removeItems("stick", 2);
            ItemStack pickaxe = new ItemStack(Items.WOODEN_PICKAXE);
            player.getInventory().offerOrDrop(pickaxe);
        } else if (targetItem.contains("axe") && hasItem("planks", 3) && hasItem("stick", 2)) {
            removeItems("planks", 3);
            removeItems("stick", 2);
            ItemStack axe = new ItemStack(Items.WOODEN_AXE);
            player.getInventory().offerOrDrop(axe);
        } else if (targetItem.contains("sword") && hasItem("planks", 2) && hasItem("stick", 1)) {
            removeItems("planks", 2);
            removeItems("stick", 1);
            ItemStack sword = new ItemStack(Items.WOODEN_SWORD);
            player.getInventory().offerOrDrop(sword);
        } else if (targetItem.contains("shovel") && hasItem("planks", 1) && hasItem("stick", 2)) {
            removeItems("planks", 1);
            removeItems("stick", 2);
            ItemStack shovel = new ItemStack(Items.WOODEN_SHOVEL);
            player.getInventory().offerOrDrop(shovel);
        } else if (targetItem.contains("stick") && hasItem("planks", 2)) {
            removeItems("planks", 2);
            ItemStack sticks = new ItemStack(Items.STICK, 4);
            player.getInventory().offerOrDrop(sticks);
        } else if (targetItem.contains("crafting_table") && hasItem("planks", 4)) {
            removeItems("planks", 4);
            ItemStack table = new ItemStack(Items.CRAFTING_TABLE);
            player.getInventory().offerOrDrop(table);
        } else if (targetItem.contains("furnace") && hasItem("cobblestone", 8)) {
            removeItems("cobblestone", 8);
            ItemStack furnace = new ItemStack(Items.FURNACE);
            player.getInventory().offerOrDrop(furnace);
        }
    }

    private void removeItems(String itemName, int count) {
        PlayerInventory inv = player.getInventory();
        int remaining = count;
        for (int i = 0; i < inv.size() && remaining > 0; i++) {
            ItemStack stack = inv.getStack(i);
            Identifier id = Registries.ITEM.getId(stack.getItem());
            if (id.getPath().contains(itemName) || id.toString().contains(itemName)) {
                int toRemove = Math.min(remaining, stack.getCount());
                stack.decrement(toRemove);
                remaining -= toRemove;
            }
        }
    }

    private void trySmelt() {
    }

    private void eatFood() {
        if (player == null) return;
        PlayerInventory inv = player.getInventory();
        for (int i = 0; i < inv.size(); i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty() && stack.getItem().isFood()) {
                inv.selectedSlot = i < 9 ? i : inv.selectedSlot;
                player.setCurrentHand(Hand.MAIN_HAND);
                break;
            }
        }
    }

    public BlockPos getBlockPos() {
        return player != null ? player.getBlockPos() : BlockPos.ORIGIN;
    }

    public Vec3d getTargetPosition() {
        return targetPosition;
    }

    public void setTargetPosition(Vec3d target) {
        this.targetPosition = target;
    }
}
