package com.gameaisystem.minecraftmod;

import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.registry.Registries;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.*;

enum Goal {
    NONE,
    GATHER_WOOD,
    CRAFT_TOOLS,
    BUILD_SHELTER,
    MINE_STONE,
    MINE_IRON,
    MINE_DIAMOND,
    FIND_FOOD,
    CRAFT_ARMOR,
    EXPLORE,
    FIND_LAVA,
    BUILD_PORTAL,
    ENTER_NETHER,
    FIND_FORTRESS,
    KILL_BLAZE,
    FIND_END_PORTAL,
    KILL_ENDER_DRAGON
}

class GoalPlanner {
    private final AIPlayer aiPlayer;
    private final Random random = new Random();
    private final Deque<Goal> goalQueue = new ArrayDeque<>();
    private Goal currentSubGoal = null;
    private int goalTimer = 0;
    private BlockPos targetBlock = null;
    private int stuckTimer = 0;
    private BlockPos lastPos = BlockPos.ORIGIN;
    private boolean waitingForDrop = false;
    private int waitTimer = 0;

    public GoalPlanner(AIPlayer aiPlayer) {
        this.aiPlayer = aiPlayer;
    }

    public void reset() {
        goalQueue.clear();
        currentSubGoal = null;
        goalTimer = 0;
        targetBlock = null;
        stuckTimer = 0;
    }

    public void startMining() {
        BlockPos pos = findBestMiningSpot();
        if (pos != null) {
            targetBlock = pos;
        }
    }

    public boolean needsMovement() {
        return targetBlock != null || aiPlayer.getTargetPosition() != null;
    }

    public void tick() {
        if (aiPlayer.getPlayer() == null) return;
        goalTimer++;

        detectStuck();

        BlockPos currentPos = aiPlayer.getBlockPos();

        if (targetBlock != null) {
            double dist = targetBlock.getSquaredDistance(currentPos);
            if (dist < 4) {
                if (aiPlayer.breakBlock(targetBlock)) {
                    collectNearbyItems();
                    targetBlock = null;
                    waitingForDrop = true;
                    waitTimer = 0;
                }
                return;
            }
            moveToward(targetBlock);
            return;
        }

        if (waitingForDrop) {
            waitTimer++;
            if (waitTimer > 40) {
                collectNearbyItems();
                waitingForDrop = false;
                waitTimer = 0;
            }
            return;
        }

        if (stuckTimer > 100) {
            unstuck();
            stuckTimer = 0;
            return;
        }

        executeGoalStep();
    }

    private void executeGoalStep() {
        Goal goal = aiPlayer.getCurrentGoal() != null ?
            Goal.valueOf(aiPlayer.getCurrentGoal()) : Goal.GATHER_WOOD;

        switch (goal) {
            case GATHER_WOOD -> gatherWood();
            case CRAFT_TOOLS -> craftTools();
            case BUILD_SHELTER -> buildShelter();
            case MINE_STONE -> mineStone();
            case MINE_IRON -> mineIron();
            case MINE_DIAMOND -> mineDiamond();
            case FIND_FOOD -> findFood();
            case CRAFT_ARMOR -> craftArmor();
            case EXPLORE -> explore();
            case FIND_LAVA -> findLava();
            case BUILD_PORTAL -> buildPortal();
            case ENTER_NETHER -> enterNether();
            case FIND_FORTRESS -> findFortress();
            case KILL_BLAZE -> killBlaze();
            case FIND_END_PORTAL -> findEndPortal();
            case KILL_ENDER_DRAGON -> killEnderDragon();
            default -> gatherWood();
        }
    }

    private void gatherWood() {
        if (goalTimer % 60 == 0) {
            BlockPos tree = findNearestBlock(new String[]{"log", "_wood"});
            if (tree != null) {
                aiPlayer.selectBestTool(aiPlayer.getWorld().getBlockState(tree));
                targetBlock = tree;
                goalTimer = 0;
            } else {
                aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(
                    (random.nextDouble() - 0.5) * 40, 0, (random.nextDouble() - 0.5) * 40
                ));
            }
        }
        if (aiPlayer.hasItem("planks", 1) || aiPlayer.hasItem("log", 8)) {
            aiPlayer.executeDirectAction("CRAFT:crafting_table");
            aiPlayer.setGoal("CRAFT_TOOLS");
            reset();
        }
    }

    private void craftTools() {
        if (!aiPlayer.hasItem("crafting_table", 1) && aiPlayer.hasItem("planks", 4)) {
            aiPlayer.executeDirectAction("CRAFT:crafting_table");
        }
        if (aiPlayer.hasItem("planks", 1) && !aiPlayer.hasItem("stick", 2)) {
            aiPlayer.hasItem("planks", 2);
            aiPlayer.executeDirectAction("CRAFT:stick");
        }
        if (!aiPlayer.hasItem("pickaxe", 1) && aiPlayer.hasItem("planks", 3) && aiPlayer.hasItem("stick", 2)) {
            aiPlayer.executeDirectAction("CRAFT:pickaxe");
        }
        if (!aiPlayer.hasItem("axe", 1) && aiPlayer.hasItem("planks", 3) && aiPlayer.hasItem("stick", 2)) {
            aiPlayer.executeDirectAction("CRAFT:axe");
        }
        if (!aiPlayer.hasItem("sword", 1) && aiPlayer.hasItem("planks", 2) && aiPlayer.hasItem("stick", 1)) {
            aiPlayer.executeDirectAction("CRAFT:sword");
        }
        if (aiPlayer.hasItem("pickaxe", 1) && aiPlayer.hasItem("axe", 1)) {
            aiPlayer.setGoal("MINE_STONE");
            reset();
        }
    }

    private void mineStone() {
        if (aiPlayer.hasItem("cobblestone", 3) || aiPlayer.hasItem("stone", 3)) {
            aiPlayer.setGoal("FIND_FOOD");
            reset();
            return;
        }
        if (goalTimer % 40 == 0) {
            BlockPos stone = findNearestBlock(new String[]{"stone", "cobblestone"});
            if (stone != null) {
                aiPlayer.selectBestTool(aiPlayer.getWorld().getBlockState(stone));
                targetBlock = stone;
            } else {
                BlockPos below = aiPlayer.getBlockPos().down((random.nextInt(3) + 1));
                if (!aiPlayer.getWorld().getBlockState(below).isAir()) {
                    targetBlock = below;
                }
            }
        }
    }

    private void mineIron() {
        if (aiPlayer.hasItem("iron_ingot", 3)) {
            aiPlayer.setGoal("MINE_DIAMOND");
            reset();
            return;
        }
        BlockPos ironOre = findNearestBlock(new String[]{"iron_ore"});
        if (ironOre != null) {
            targetBlock = ironOre;
        } else {
            aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(0, -5, 0));
        }
    }

    private void mineDiamond() {
        if (aiPlayer.hasItem("diamond", 5)) {
            aiPlayer.setGoal("FIND_LAVA");
            reset();
            return;
        }
        int y = aiPlayer.getBlockPos().getY();
        if (y > -54) {
            aiPlayer.setTargetPosition(new Vec3d(
                aiPlayer.getBlockPos().getX(), -54, aiPlayer.getBlockPos().getZ()
            ));
        }
        BlockPos diamond = findNearestBlock(new String[]{"diamond_ore"});
        if (diamond != null) {
            targetBlock = diamond;
        } else {
            aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(
                (random.nextDouble() - 0.5) * 10, 0, (random.nextDouble() - 0.5) * 10
            ));
        }
    }

    private void findFood() {
        BlockPos food = findNearestBlock(new String[]{"wheat", "carrots", "potatoes", "_berry"});
        if (food != null) {
            targetBlock = food;
        } else if (aiPlayer.hasItem("wheat_seeds", 1)) {
            BlockPos ground = findNearestBlock(new String[]{"grass_block", "dirt"});
            if (ground != null) {
                aiPlayer.placeBlock(ground.up(), new ItemStack(Items.WHEAT_SEEDS));
            }
        } else {
            aiPlayer.executeDirectAction("MOVE:" +
                (aiPlayer.getBlockPos().getX() + (random.nextDouble() - 0.5) * 30) + ":0:" +
                (aiPlayer.getBlockPos().getZ() + (random.nextDouble() - 0.5) * 30));
        }
        if (aiPlayer.hasItem("bread", 3) || aiPlayer.hasItem("apple", 3) ||
            aiPlayer.hasItem("cooked_beef", 3) || aiPlayer.hasItem("cooked_porkchop", 3)) {
            aiPlayer.setGoal("MINE_IRON");
            reset();
        }
    }

    private void buildShelter() {
        if (aiPlayer.hasItem("planks", 10)) {
            BlockPos base = aiPlayer.getBlockPos();
            for (int y = 0; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    for (int z = -2; z <= 2; z++) {
                        if (x != 0 || z != 0 || y == 2) {
                        }
                    }
                }
            }
            aiPlayer.setGoal("MINE_STONE");
            reset();
        }
    }

    private void explore() {
        if (goalTimer % 200 == 0) {
            aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(
                (random.nextDouble() - 0.5) * 200, 0, (random.nextDouble() - 0.5) * 200
            ));
        }
    }

    private void findLava() {
        BlockPos lava = findNearestBlock(new String[]{"lava"});
        if (lava != null) {
            targetBlock = lava;
            aiPlayer.setGoal("BUILD_PORTAL");
        } else {
            aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(0, -10, 0));
        }
    }

    private void buildPortal() {
        aiPlayer.setGoal("ENTER_NETHER");
    }

    private void enterNether() {
        BlockPos portal = findNearestBlock(new String[]{"nether_portal"});
        if (portal != null) {
            targetBlock = portal;
        }
    }

    private void findFortress() {
        aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(
            (random.nextDouble() - 0.5) * 200, 0, (random.nextDouble() - 0.5) * 200
        ));
    }

    private void killBlaze() {
        List<net.minecraft.entity.mob.HostileEntity> hostiles = aiPlayer.getNearbyHostiles();
        for (net.minecraft.entity.mob.HostileEntity entity : hostiles) {
            if (Registries.ENTITY_TYPE.getId(entity.getType()).getPath().equals("blaze")) {
                targetBlock = entity.getBlockPos();
                return;
            }
        }
    }

    private void findEndPortal() {
        BlockPos endPortal = findNearestBlock(new String[]{"end_portal_frame"});
        if (endPortal != null) {
            targetBlock = endPortal;
            aiPlayer.setGoal("KILL_ENDER_DRAGON");
        }
    }

    private void killEnderDragon() {
        List<net.minecraft.entity.mob.HostileEntity> hostiles = aiPlayer.getNearbyHostiles();
        for (net.minecraft.entity.mob.HostileEntity entity : hostiles) {
            String typeName = Registries.ENTITY_TYPE.getId(entity.getType()).getPath();
            if (typeName.equals("ender_dragon")) {
                targetBlock = entity.getBlockPos();
                return;
            }
        }
    }

    private BlockPos findBestMiningSpot() {
        BlockPos playerPos = aiPlayer.getBlockPos();
        int y = Math.min(playerPos.getY(), -54);
        return playerPos.withY(y - 5);
    }

    private BlockPos findNearestBlock(String[] blockIds) {
        BlockPos origin = aiPlayer.getBlockPos();
        BlockPos best = null;
        double bestDist = Double.MAX_VALUE;
        int searchRadius = 32;

        BlockPos.Mutable mbp = new BlockPos.Mutable();
        for (int y = -12; y <= 12; y++) {
            for (int x = -searchRadius; x <= searchRadius; x++) {
                for (int z = -searchRadius; z <= searchRadius; z++) {
                    mbp.set(origin.getX() + x, origin.getY() + y, origin.getZ() + z);
                    String blockId = Registries.BLOCK.getId(
                        aiPlayer.getWorld().getBlockState(mbp).getBlock()
                    ).toString().toLowerCase();
                    for (String target : blockIds) {
                        if (blockId.contains(target)) {
                            double dist = mbp.getSquaredDistance(origin);
                            if (dist < bestDist) {
                                bestDist = dist;
                                best = mbp.toImmutable();
                            }
                        }
                    }
                }
            }
        }
        return best;
    }

    private void moveToward(BlockPos target) {
        Vec3d targetCenter = Vec3d.ofCenter(target);
        aiPlayer.setTargetPosition(targetCenter);
    }

    private void collectNearbyItems() {
        List<net.minecraft.entity.ItemEntity> items = aiPlayer.getNearbyItems(4);
        for (net.minecraft.entity.ItemEntity item : items) {
            aiPlayer.setTargetPosition(item.getPos());
        }
    }

    private void detectStuck() {
        BlockPos current = aiPlayer.getBlockPos();
        if (current.equals(lastPos)) {
            stuckTimer++;
        } else {
            stuckTimer = 0;
            lastPos = current;
        }
    }

    private void unstuck() {
        aiPlayer.setTargetPosition(aiPlayer.getBlockPos().add(
            (random.nextDouble() - 0.5) * 10,
            random.nextDouble() * 3,
            (random.nextDouble() - 0.5) * 10
        ));
    }
}
