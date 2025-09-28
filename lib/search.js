// POST /api/search  { query, k?, doc_id? }
import { embedText } from "../lib/embeddings.js";
import { supabase } from "../lib/supabase.js";

const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { query, k = 5, doc_id } = req.body || {};
    if (!query) return res.status(400).json({ error: "missing query" });

    const qvec = await embedText(query);
    const { data, error } = await supabase.rpc("match_documents", {
      query_embedding: qvec,
      match_count: k * 3  // 多取一點再過濾
    });
    if (error) throw error;

    let results = (data || []).map(r => ({
      id: r.id,
      title: r.title,
      text: r.content,
      score: r.similarity
    }));

    // 可選：只要特定文件
    if (doc_id) results = results.filter(r => r.id.startsWith(`${doc_id}:`));

    // 相似度門檻，避免亂入
    const MIN_SIM = 0.75;
    results = results.filter(r => r.score >= MIN_SIM).slice(0, k);

    res.json(results);
  } catch (e) {
    res.status(500).json({ error: e.message || "search failed" });
  }
}
