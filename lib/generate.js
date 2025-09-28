// POST /api/generate  { query, contexts: [{text, id?, title?, score?}] }
import { openai } from "../lib/embeddings.js";

const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { query, contexts = [] } = req.body || {};
    if (!query) return res.status(400).json({ error: "missing query" });

    // 打包上下文（限制長度，避免超 token）
    let used = 0, packed = [];
    for (const c of contexts) {
      const t = (c.text || "").trim();
      if (!t) continue;
      if (used + t.length > 8000) break;
      packed.push(t); used += t.length;
    }
    const contextText = packed.map((t,i)=>`[${i+1}] ${t}`).join("\n\n");

    const system = `You are a strict RAG assistant.
- Answer ONLY with the provided context snippets; if info is missing, reply "Insufficient information."
- Keep a neutral, diplomatic tone; avoid blunt judgments like "not suitable."
- Use inline citations [1], [2], ... referring to the snippet order.`;

    const user = `Question: ${query}\n\nContext:\n${contextText}`;

    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ]
    });

    const text = r.choices?.[0]?.message?.content ?? "";
    const citations = packed.map((_, i) => ({ id: String(i+1), title: `Source ${i+1}` }));
    res.json({ text, citations });
  } catch (e) {
    res.status(500).json({ error: e.message || "generate failed" });
  }
}