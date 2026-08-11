package com.gameaisystem.minecraftmod;

import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.ItemStack;
import net.minecraft.item.SwordItem;
import net.minecraft.registry.Registries;
import net.minecraft.util.Hand;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

class CombatController {
    private static final Logger LOGGER = LoggerFactory.getLogger("CombatController");
    private final AIPlayer aiPlayer;
    private boolean inCombat = false;
    private HostileEntity currentTarget = null;
    private int combatTimer = 0;
    private int attackCooldown = 0;

    public CombatController(AIPlayer aiPlayer) {
        this.aiPlayer = aiPlayer;
    }

    public boolean isInCombat() { return inCombat; }

    public void checkThreats() {
        if (aiPlayer.getPlayer() == null) return;

        List<HostileEntity> hostiles = aiPlayer.getNearbyHostiles();
        if (!hostiles.isEmpty()) {
            inCombat = true;
            double closestDist = Double.MAX_VALUE;
            for (HostileEntity e : hostiles) {
                double dist = e.squaredDistanceTo(aiPlayer.getPlayer());
                if (dist < closestDist) {
                    closestDist = dist;
                    currentTarget = e;
                }
            }

            equipBestWeapon();
            aiPlayer.setTargetPosition(currentTarget.getPos());
        } else {
            inCombat = false;
            currentTarget = null;
        }
    }

    public void tick() {
        if (!inCombat || currentTarget == null || aiPlayer.getPlayer() == null) return;

        combatTimer++;
        if (attackCooldown > 0) attackCooldown--;

        if (currentTarget.isDead() || !currentTarget.isAlive()) {
            inCombat = false;
            currentTarget = null;
            return;
        }

        double dist = aiPlayer.getPlayer().squaredDistanceTo(currentTarget);
        if (dist > 49) {
            aiPlayer.setTargetPosition(currentTarget.getPos());
        }

        if (dist < 9 && attackCooldown <= 0) {
            aiPlayer.getPlayer().attack(currentTarget);
            aiPlayer.getPlayer().swingHand(Hand.MAIN_HAND);
            attackCooldown = 10;
        }
    }

    public void attackNearest() {
        List<HostileEntity> hostiles = aiPlayer.getNearbyHostiles();
        if (!hostiles.isEmpty()) {
            HostileEntity target = hostiles.get(0);
            if (aiPlayer.getPlayer() != null) {
                aiPlayer.getPlayer().attack(target);
                aiPlayer.getPlayer().swingHand(Hand.MAIN_HAND);
            }
        }
    }

    private void equipBestWeapon() {
        if (aiPlayer.getPlayer() == null) return;
        PlayerInventory inv = aiPlayer.getPlayer().getInventory();
        int bestSlot = inv.selectedSlot;
        float bestDamage = 0;

        for (int i = 0; i < 9; i++) {
            ItemStack stack = inv.getStack(i);
            if (!stack.isEmpty() && stack.getItem() instanceof SwordItem sword) {
                float damage = sword.getAttackDamage();
                if (damage > bestDamage) {
                    bestDamage = damage;
                    bestSlot = i;
                }
            }
        }
        if (bestSlot != inv.selectedSlot) {
            inv.selectedSlot = bestSlot;
        }
    }
}
