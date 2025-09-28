import { embedText } from "../lib/embeddings.js";
import { supabase } from "../lib/supabase.js";

const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

async function readJson(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const raw = Buffer.concat(chunks).toString("utf8");
  try { return raw ? JSON.parse(raw) : {}; } catch { return {}; }
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).end();

  try {
    if (!process.env.OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");
    if (!process.env.SUPABASE_URL) throw new Error("Missing SUPABASE_URL");
    if (!process.env.SUPABASE_SERVICE_ROLE) throw new Error("Missing SUPABASE_SERVICE_ROLE");

    const body = await readJson(req);
    const { query, k = 5, doc_id } = body || {};
    if (!query) return res.status(400).json({ error: "missing query" });

    const qvec = await embedText(query);
    const { data, error } = await supabase.rpc("match_documents", {
      query_embedding: qvec,
      match_count: Math.max(5, k * 3)
    });
    if (error) throw error;

    let results = (data || []).map(r => ({
      id: r.id,
      title: r.title,
      text: r.content,
      score: r.similarity
    }));

    if (doc_id) results = results.filter(r => r.id.startsWith(`${doc_id}:`));

    const MIN_SIM = 0.75;
    results = results.filter(r => r.score >= MIN_SIM).slice(0, k);

    res.setHeader("content-type", "application/json; charset=utf-8");
    return res.status(200).end(JSON.stringify(results));
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
