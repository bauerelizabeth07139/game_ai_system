package com.gameaisystem.minecraftmod;

import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.*;
import net.minecraft.registry.Registries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class InventoryManager {
    private static final Logger LOGGER = LoggerFactory.getLogger("InventoryManager");
    private final AIPlayer aiPlayer;

    public InventoryManager(AIPlayer aiPlayer) {
        this.aiPlayer = aiPlayer;
    }

    public void organize() {
        if (aiPlayer.getPlayer() == null) return;
        PlayerInventory inv = aiPlayer.getPlayer().getInventory();

        for (int i = 0; i < 9; i++) {
            ItemStack hotbarStack = inv.getStack(i);
            if (!hotbarStack.isEmpty() && isTool(hotbarStack)) continue;

            for (int j = 9; j < inv.size(); j++) {
                ItemStack mainStack = inv.getStack(j);
                if (!mainStack.isEmpty() && isTool(mainStack)) {
                    inv.setStack(j, hotbarStack.copy());
                    inv.setStack(i, mainStack.copy());
                    break;
                }
            }
        }
    }

    public void eatIfHungry() {
        if (aiPlayer.getPlayer() == null) return;
        if (aiPlayer.getPlayer().getHungerManager().getFoodLevel() < 15) {
            PlayerInventory inv = aiPlayer.getPlayer().getInventory();
            for (int i = 0; i < inv.size(); i++) {
                ItemStack stack = inv.getStack(i);
                if (!stack.isEmpty() && stack.getItem().isFood()) {
                    inv.selectedSlot = i < 9 ? i : inv.selectedSlot;
                    if (i >= 9) {
                        ItemStack temp = inv.getStack(inv.selectedSlot);
                        inv.setStack(inv.selectedSlot, stack.copy());
                        inv.setStack(i, temp.copy());
                    }
                    break;
                }
            }
        }
    }

    public void dropJunk() {
        if (aiPlayer.getPlayer() == null) return;
        PlayerInventory inv = aiPlayer.getPlayer().getInventory();
        for (int i = 9; i < inv.size(); i++) {
            ItemStack stack = inv.getStack(i);
            String itemName = Registries.ITEM.getId(stack.getItem()).getPath();
            if (itemName.contains("dirt") || itemName.contains("gravel") ||
                itemName.contains("sand") || itemName.contains("rotten_flesh") ||
                itemName.contains("spider_eye") || itemName.contains("bone")) {
                inv.removeStack(i);
            }
        }
    }

    private boolean isTool(ItemStack stack) {
        Item item = stack.getItem();
        return item instanceof SwordItem || item instanceof PickaxeItem ||
            item instanceof AxeItem || item instanceof ShovelItem ||
            item instanceof HoeItem;
    }
}
