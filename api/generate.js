// api/generate.js  — no inline citations; adds simple "see Research/Projects/GitHub" hint.
import OpenAI from "openai";
import { embedText } from "../lib/embeddings.js";
import { supabase } from "../lib/supabase.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

// Robust JSON reader (avoid relying on req.body)
async function readJson(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const raw = Buffer.concat(chunks).toString("utf8");
  try { return raw ? JSON.parse(raw) : {}; } catch { return {}; }
}

// Heuristics to suggest where to read more
function computeHints(ctxTexts) {
  const L = ctxTexts.join("\n\n").toLowerCase();
  const hasResearch =
    /research|lab|laboratory|wrhs|chen lab|computational human|neuroscience|taiwanese psychological association|manuscript/.test(L);
  const hasProjects =
    /project|poster|conference|lstm|imputation|argument mining|water quality|presentation/.test(L);
  const hasGithub = /github\.com|edchang716/.test(L);
  const hints = [];
  if (hasResearch) hints.push("Research section");
  if (hasProjects) hints.push("Projects section");
  if (hasGithub) hints.push("GitHub profile");
  return hints;
}

export default async function handler(req, res) {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { query, contexts, k = 12, temperature = 0.2 } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 1) If contexts not provided, self-retrieve
    let ctx = Array.isArray(contexts) ? contexts : [];
    if (!ctx.length) {
      const qvec = await embedText(query);
      const { data, error } = await supabase.rpc("match_documents", {
        query_embedding: qvec,
        match_count: Math.max(5, k * 3)
      });
      if (error) throw error;
      ctx = (data || []).map(r => ({
        id: r.id,
        title: r.title,
        text: r.content,           // RPC returns content
        score: r.similarity,
        metadata: r.metadata
      }));
    }

    // 2) Clean & cap contexts (NO citations will be inserted)
    const usable = ctx
      .map(c => (c.text || c.content || c.chunk || "").toString().trim())
      .filter(Boolean)
      .slice(0, k);

    if (!usable.length) {
      return res.status(200).json({ text: "Insufficient information." });
    }

    // 3) Pack context with a generous limit (character-based)
    let used = 0, packed = [];
    const LIMIT = 14000;
    for (const t of usable) {
      if (used + t.length > LIMIT) break;
      packed.push(t);
      used += t.length;
    }
    const contextBlob = packed.join("\n\n---\n\n");

    // 4) Generation prompt — STRICT to context, NO citations
    const system = `You are a strict RAG assistant.
- Answer ONLY using the provided context below.
- If the information is not present, reply "Insufficient information."
- Be concise, neutral, and professional. Do NOT include bracketed citations or source IDs.`;

    const user = `Question: ${query}

Context:
${contextBlob}`;

    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ]
    });

    let text = r.choices?.[0]?.message?.content?.trim() || "Insufficient information.";

    // 5) Append a simple hint for where to learn more (based on matched context)
    const hints = computeHints(packed);
    if (hints.length) {
      text += `\n\nFor more details, see: ${hints.join(", ")}.`;
    }

    res.setHeader("content-type", "application/json; charset=utf-8");
    return res.status(200).end(JSON.stringify({ text }));
  } catch (e) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
}
