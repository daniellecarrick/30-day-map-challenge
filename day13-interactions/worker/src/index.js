/**
 * Day 13 · Interactions — "Can you draw these countries?" drawing store.
 *
 *   POST /drawing  {country, ring, seq, session}  -> saves one visitor's outline of a country
 *   GET  /drawings?country=UKR                    -> every saved outline of that country, for the heat map
 *   GET  /counts                                  -> how many outlines each country has
 *
 * No API keys: this Worker only stores shapes. Rings are [[lon, lat], ...] in degrees, drawn on an
 * Equal Earth map. The page does all scoring; build.py recomputes it for the PNG.
 */

// The countries a visitor can be asked to draw (Natural Earth ADM0_A3 codes). Keep in step with build.py.
const COUNTRIES = new Set([
  "RUS", "CAN", "BRA", "COD", "MEX",   // size test
  "UKR", "IRN", "VEN", "AFG",          // in US news
  "NER",                               // often confused with Nigeria
]);
const MAX_POINTS = 250;
const MAX_AREA_KM2 = 100e6;   // Earth's land is ~149M km²; anything this big isn't an attempt at one country
const MIN_AREA_KM2 = 500;     // smaller than a city: a mis-tap, not a drawing

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const cors = corsHeaders(req, env);
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    try {
      if (req.method === "POST" && url.pathname === "/drawing") return json(await save(req, env), 200, cors);
      if (req.method === "GET" && url.pathname === "/drawings")
        return json(await drawings(env, url.searchParams.get("country")), 200, { ...cors, "Cache-Control": "public, max-age=60" });
      if (req.method === "GET" && url.pathname === "/counts")
        return json(await counts(env), 200, { ...cors, "Cache-Control": "public, max-age=60" });
      return json({ error: "Not found" }, 404, cors);
    } catch (e) {
      if (e instanceof HttpError) {
        const h = e.retryAfter ? { ...cors, "Retry-After": String(e.retryAfter) } : cors;
        return json({ error: e.message, code: e.code, retry_after: e.retryAfter }, e.status, h);
      }
      console.error(e);
      return json({ error: "Something went wrong on the server." }, 500, cors);
    }
  },
};

class HttpError extends Error {
  constructor(status, message, { code = "error", retryAfter } = {}) {
    super(message); this.status = status; this.code = code; this.retryAfter = retryAfter;
  }
}

function corsHeaders(req, env) {
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
  const origin = req.headers.get("Origin");
  const h = { "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type", Vary: "Origin" };
  if (origin && allowed.includes(origin)) h["Access-Control-Allow-Origin"] = origin;
  return h;
}

const json = (body, status = 200, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

async function readJson(req) {
  try { return await req.json(); } catch { throw new HttpError(400, "Send a JSON body."); }
}

async function ipHash(req, env) {
  // Salted and truncated: enough to rate-limit one visitor, not enough to recover the address.
  const ip = req.headers.get("CF-Connecting-IP") || "unknown";
  const day = new Date().toISOString().slice(0, 10);   // the salt changes daily, so hashes can't be linked across days
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(ip + day + (env.IP_SALT || "day13")));
  return [...new Uint8Array(buf)].slice(0, 12).map(b => b.toString(16).padStart(2, "0")).join("");
}

// Area of a lon/lat ring in km² (planar shoelace on cos-scaled degrees: rough, but only used for sanity limits)
function areaKm2(ring) {
  const r = Math.PI / 180, k = 111.32;
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [x0, y0] = ring[j], [x1, y1] = ring[i];
    a += (x0 * Math.cos(y0 * r)) * y1 - (x1 * Math.cos(y1 * r)) * y0;
  }
  return Math.abs(a / 2) * k * k;
}

function cleanRing(raw) {
  if (!Array.isArray(raw) || raw.length < 4 || raw.length > MAX_POINTS)
    throw new HttpError(400, `A drawing needs between 4 and ${MAX_POINTS} points.`);
  const ring = raw.map(p => {
    const lon = Number(p?.[0]), lat = Number(p?.[1]);
    if (!(lon >= -180 && lon <= 180 && lat >= -90 && lat <= 90)) throw new HttpError(400, "A point in the drawing is off the map.");
    return [Math.round(lon * 100) / 100, Math.round(lat * 100) / 100];
  });
  const a = areaKm2(ring);
  if (a < MIN_AREA_KM2) throw new HttpError(400, "That shape is too small to count. Try drawing it bigger.");
  if (a > MAX_AREA_KM2) throw new HttpError(400, "That shape covers most of the world. Try again a bit smaller.");
  return ring;
}

// ---------- POST /drawing ----------
async function save(req, env) {
  const body = await readJson(req);
  const country = String(body.country || "").toUpperCase();
  if (!COUNTRIES.has(country)) throw new HttpError(400, "That country isn't part of this game.");
  const ring = cleanRing(body.ring);
  const seq = Math.max(1, Math.min(20, Math.round(Number(body.seq)) || 1));     // which drawing of the round (1 = first)
  const session = /^[a-z0-9-]{8,40}$/i.test(body.session || "") ? body.session : null;

  const hash = await ipHash(req, env);
  const now = Math.floor(Date.now() / 1000);
  const hourly = Number(env.HOURLY_LIMIT || 60), daily = Number(env.DAILY_LIMIT || 20000);
  const { h, oldest, d } = await env.DB.prepare(
    `SELECT SUM(ip_hash = ?1 AND created_at > ?2) AS h,
            MIN(CASE WHEN ip_hash = ?1 AND created_at > ?2 THEN created_at END) AS oldest,
            COUNT(*) AS d
     FROM drawings WHERE created_at > ?3`
  ).bind(hash, now - 3600, now - 86400).first();
  if ((h || 0) >= hourly) {
    const wait = Math.max(60, (oldest || now) + 3600 - now), mins = Math.ceil(wait / 60);
    throw new HttpError(429, `You've drawn a lot this hour, so new drawings aren't being saved. Try again in ${mins} minute${mins === 1 ? "" : "s"}.`,
      { code: "rate_limit", retryAfter: wait });
  }
  if ((d || 0) >= daily) throw new HttpError(429, "The map has collected all the drawings it can take today. Come back tomorrow.", { code: "daily_limit" });

  const id = crypto.randomUUID();
  await env.DB.prepare(
    "INSERT INTO drawings (id, created_at, ip_hash, country, ring, n_points, seq, session) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"
  ).bind(id, now, hash, country, JSON.stringify(ring), ring.length, seq, session).run();
  return { saved: true, id };
}

// ---------- GET /drawings ----------
async function drawings(env, raw) {
  const country = String(raw || "").toUpperCase();
  if (!COUNTRIES.has(country)) throw new HttpError(400, "Ask for one of the game's countries, e.g. ?country=UKR.");
  const { results } = await env.DB.prepare(
    "SELECT id, ring, seq, created_at FROM drawings WHERE country = ?1 AND hidden = 0 ORDER BY created_at DESC LIMIT 3000"
  ).bind(country).all();
  return { country, drawings: results.map(r => ({ id: r.id, ring: JSON.parse(r.ring), seq: r.seq, at: r.created_at })) };
}

// ---------- GET /counts ----------
async function counts(env) {
  const { results } = await env.DB.prepare("SELECT country, COUNT(*) AS n FROM drawings WHERE hidden = 0 GROUP BY country").all();
  return { counts: Object.fromEntries(results.map(r => [r.country, r.n])) };
}
