package com.gameaisystem.minecraftmod;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;
import net.minecraft.client.render.entity.EntityRendererFactory;
import net.minecraft.client.render.entity.PlayerEntityRenderer;

public class MinecraftAIModClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        EntityRendererRegistry.register(ModEntities.FAKE_PLAYER,
            (EntityRendererFactory.Context context) -> new PlayerEntityRenderer(context, false));
    }
}
