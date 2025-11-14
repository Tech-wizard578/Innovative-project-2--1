const API = import.meta.env.VITE_API_BASE;

async function request(method: string, path: string, data?: any) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: data ? JSON.stringify(data) : undefined
  });

  const json = await res.json();
  if (!json.success) throw new Error(json.error || "Unknown API error");
  return json;
}

export const api = {
  presets: () => request("GET", "/api/presets"),
  schedule: (payload: any) => request("POST", "/api/schedule", payload),
  compare: (payload: any) => request("POST", "/api/compare", payload),
  explain: (payload: any) => request("POST", "/api/explain", payload),
  energy: (payload: any) => request("POST", "/api/energy-metrics", payload),
  report: (payload: any) => request("POST", "/api/export", payload)
};
