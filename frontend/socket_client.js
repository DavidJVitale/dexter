export function createSocketClient() {
  if (!window.io) {
    throw new Error("Socket.IO client library not loaded");
  }
  const backendOrigin = `${window.location.protocol}//${window.location.hostname}:5000`;
  return window.io(backendOrigin, {
    transports: ["polling"],
  });
}
