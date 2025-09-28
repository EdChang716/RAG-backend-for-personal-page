// POST /api/generate  { query, contexts: [{text, id?, title?, score?}] }
// api/generate.js
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import { embedText } from "../lib/embeddings.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE);
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

async function readJson(req) {
  const chunks = []; for await (const c of req) chunks.push(c);
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
    const { query, contexts = [], k = 12 } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 如果沒給 contexts，就自己查一次
    let ctx = contexts;
    if (!ctx?.length) {
      const qvec = await embedText(query);
      const { data, error } = await supabase.rpc("match_documents", {
        query_embedding: qvec,
        match_count: Math.max(5, k * 3)
      });
      if (error) throw error;
      ctx = (data || []).map(r => ({ id: r.id, title: r.title, text: r.content, score: r.similarity, metadata: r.metadata }));
    }

    // 過濾空白、打包長度（加大一點上限）
    const usable = (ctx || []).filter(c => (c.text||"").trim().length > 0).slice(0, k);
    if (!usable.length) return res.status(200).json({ text: "Insufficient information.", citations: [] });

    let used = 0, packed = [];
    const LIMIT = 14000; // 比原本 8000 寬鬆
    for (const c of usable) {
      const t = c.text.trim();
      if (used + t.length > LIMIT) break;
      packed.push(t); used += t.length;
    }
    const numbered = packed.map((t,i)=>`[${i+1}] ${t}`).join("\n\n");

    const system = `You are a strict RAG assistant.
- Answer ONLY using the provided context snippets; if info is missing, reply "Insufficient information."
- Be concise, neutral, and diplomatic (avoid blunt judgments).
- Use inline citations [1], [2] referencing snippet order.`;

    const user = `Question: ${query}\n\nContext snippets:\n${numbered}`;

    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      messages: [{ role: "system", content: system }, { role: "user", content: user }]
    });

    const text = r.choices?.[0]?.message?.content ?? "";
    const citations = usable.map((c, i) => ({
      id: String(i+1),
      title: c.title || `Source ${i+1}`,
      score: c.score,
      metadata: c.metadata
    }));

    res.setHeader("content-type", "application/json; charset=utf-8");
    return res.status(200).end(JSON.stringify({ text, citations }));
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
