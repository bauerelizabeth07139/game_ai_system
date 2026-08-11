package com.gameaisystem.minecraftmod;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class AITCPServer {
    private static final Logger LOGGER = LoggerFactory.getLogger("AITCPServer");
    private final int port;
    private ServerSocket serverSocket;
    private Thread serverThread;
    private volatile boolean running = false;
    private final List<ClientHandler> clients = new CopyOnWriteArrayList<>();

    public AITCPServer(int port) {
        this.port = port;
    }

    public void start() {
        if (running) return;
        running = true;
        serverThread = new Thread(() -> {
            try {
                serverSocket = new ServerSocket(port);
                serverSocket.setReuseAddress(true);
                LOGGER.info("AI TCP Server listening on port {}", port);

                while (running) {
                    try {
                        Socket client = serverSocket.accept();
                        ClientHandler handler = new ClientHandler(client);
                        clients.add(handler);
                        handler.start();
                    } catch (SocketException e) {
                        if (running) LOGGER.warn("Socket exception: {}", e.getMessage());
                    } catch (IOException e) {
                        if (running) LOGGER.error("Error accepting client: {}", e.getMessage());
                    }
                }
            } catch (IOException e) {
                LOGGER.error("Failed to start TCP server: {}", e.getMessage());
            }
        }, "AI-TCP-Server");
        serverThread.setDaemon(true);
        serverThread.start();
    }

    public void stop() {
        running = false;
        for (ClientHandler client : clients) {
            client.disconnect();
        }
        clients.clear();
        try {
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
            }
        } catch (IOException e) {
            LOGGER.warn("Error closing server socket: {}", e.getMessage());
        }
        LOGGER.info("AI TCP Server stopped");
    }

    public void sendToAll(String message) {
        for (ClientHandler client : clients) {
            client.send(message);
        }
    }

    public List<ClientHandler> getClients() {
        return Collections.unmodifiableList(clients);
    }

    class ClientHandler extends Thread {
        private final Socket socket;
        private PrintWriter writer;
        private BufferedReader reader;
        private boolean connected = true;

        public ClientHandler(Socket socket) {
            this.socket = socket;
            setDaemon(true);
        }

        @Override
        public void run() {
            try {
                writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF-8"), true);
                reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), "UTF-8"));

                writer.println("{\"type\":\"connected\",\"source\":\"minecraft_ai_mod\"}");

                String line;
                while (connected && (line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    if ("STOP".equals(line)) {
                        connected = false;
                        break;
                    }
                    MinecraftAIMod.commandQueue.offer(line);
                }
            } catch (IOException e) {
                if (connected) LOGGER.warn("Client handler error: {}", e.getMessage());
            } finally {
                disconnect();
            }
        }

        public void send(String message) {
            if (writer != null && connected) {
                try {
                    writer.println(message);
                    writer.flush();
                } catch (Exception e) {
                    LOGGER.warn("Failed to send message to client: {}", e.getMessage());
                }
            }
        }

        public void disconnect() {
            connected = false;
            try {
                if (socket != null && !socket.isClosed()) {
                    socket.close();
                }
            } catch (IOException ignored) {}
            clients.remove(this);
        }
    }
}
