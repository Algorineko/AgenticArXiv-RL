import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL,
  timeout: 240000,
  headers: { "Content-Type": "application/json" },
});

export function getApiBase() {
  return baseURL;
}
