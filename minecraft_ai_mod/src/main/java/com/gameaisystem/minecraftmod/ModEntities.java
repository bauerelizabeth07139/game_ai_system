package com.gameaisystem.minecraftmod;

import net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityTypeBuilder;
import net.minecraft.entity.EntityDimensions;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnGroup;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;

public class ModEntities {
    public static final EntityType<AIPlayerEntity> FAKE_PLAYER = Registry.register(
        Registries.ENTITY_TYPE,
        new Identifier(MinecraftAIMod.MOD_ID, "fake_player"),
        FabricEntityTypeBuilder.create(SpawnGroup.MISC, AIPlayerEntity::new)
            .dimensions(EntityDimensions.fixed(0.6f, 1.8f))
            .trackRangeChunks(64)
            .trackedUpdateRate(3)
            .build()
    );

    public static void register() {
        MinecraftAIMod.LOGGER.info("Registered fake player entity");
    }
}
