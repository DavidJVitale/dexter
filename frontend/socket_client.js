export function createSocketClient() {
  if (!window.io) {
    throw new Error("Socket.IO client library not loaded");
  }
  const backendOrigin =
    window.location.port === "5173"
      ? `${window.location.protocol}//${window.location.hostname}:5050`
      : window.location.origin;
  return window.io(backendOrigin, {
    transports: ["polling"],
    withCredentials: false,
  });
}
