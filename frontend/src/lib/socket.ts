// src/lib/socket.ts
import { io } from "socket.io-client";

// This is the URL of your Python backend
const SERVER_URL = "http://127.0.0.1:8000";

export const socket = io(SERVER_URL, {
  transports: ["websocket"], // Use WebSocket transport
});