export function createSocketClient() {
  if (!window.io) {
    throw new Error("Socket.IO client library not loaded");
  }
  return window.io("http://127.0.0.1:5000", {
    transports: ["websocket", "polling"],
  });
}
