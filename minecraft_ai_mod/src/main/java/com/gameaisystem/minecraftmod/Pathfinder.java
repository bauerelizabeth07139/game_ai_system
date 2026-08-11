package com.gameaisystem.minecraftmod;

import net.minecraft.util.math.BlockPos;

import java.util.*;

class Pathfinder {
    private final Random random = new Random();

    public List<BlockPos> findPath(BlockPos from, BlockPos to) {
        List<BlockPos> path = new ArrayList<>();
        path.add(from);

        BlockPos current = from;
        while (!current.equals(to)) {
            int dx = Integer.compare(to.getX(), current.getX());
            int dy = Integer.compare(to.getY(), current.getY());
            int dz = Integer.compare(to.getZ(), current.getZ());

            if (dx != 0 && dz != 0 && random.nextBoolean()) {
                current = current.add(dx, 0, 0);
            } else if (dz != 0) {
                current = current.add(0, 0, dz);
            } else if (dx != 0) {
                current = current.add(dx, 0, 0);
            } else if (dy != 0) {
                current = current.add(0, dy, 0);
            } else {
                break;
            }
            path.add(current);
            if (path.size() > 100) break;
        }
        return path;
    }
}
