// api/generate.js
// Rules:
// - no citations
// - small talk allowed
// - fallback in EN with contact
// - hints: multi-project => generic "for more..."
//          single-project => CHEN/WaterQuality -> resume+Research
//                            others            -> GitHub+Projects

import OpenAI from "openai";
import { embedText } from "../lib/embeddings.js";
import { supabase } from "../lib/supabase.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

const FALLBACK_EN =
  "We can’t answer your question based on the knowledge base. Please try a more specific query, or contact cc5375@columbia.edu for more information.";

// ---------- utils ----------
async function readJson(req){
  const chunks=[]; for await (const c of req) chunks.push(c);
  const raw = Buffer.concat(chunks).toString("utf8");
  try{ return raw ? JSON.parse(raw) : {}; }catch{ return {}; }
}

function greetIfSmallTalk(q){
  if (!q) return null;
  const s = q.trim().toLowerCase();
  const re = /^(hi|hello|hey|yo|howdy|sup|嗨|你好|早安|午安|晚安)[!.? ]*$/i;
  if (re.test(s)) {
    return "Hi! I’m EDDi, Edward’s assistant. Ask me about his background, labs, projects, resume, or GitHub.";
  }
  return null;
}

// ---- Project detectors (keywords you can extend) ----
const PROJECT_MATCHERS = [
  { key: "WATER_QUALITY", label: "Water Quality Prediction",
    re: /(water\s+quality|lstm[-\s]?ed|imputation|taipei\s+bridge)/i },
  { key: "CHEN_LAB", label: "CHEN Lab",
    re: /(chen\s+lab|computational\s+human|neuroscience\s+lab|p(-|\.)?h\.?\s*a\.?\s*chen)/i },
  { key: "ANNA", label: "Anna Karenina / Well-being",
    re: /(anna\s*karenina|well[-\s]?being)/i },
  { key: "CROSS_DOMAIN", label: "Cross-domain Text Semantic Similarity",
    re: /(cross[-\s]?domain|semantic\s+similarity|embedding[s]?\s+bias)/i },
  { key: "ARG_MINING", label: "Argument Mining",
    re: /(argument\s+mining)/i },
  { key: "CAYIN", label: "CAYIN Generative AI (Internship)",
    re: /(cayin)/i },
];

function detectProjectsFrom(query, keptRows){
  const bag = [
    (query||""),
    ...keptRows.map(r => [r.title||"", r._t||"", r.id||"", JSON.stringify(r.metadata||{})].join("\n"))
  ].join("\n");

  const found = [];
  for (const m of PROJECT_MATCHERS){
    if (m.re.test(bag)) found.push({ key: m.key, label: m.label });
  }
  const seen = new Set();
  return found.filter(x => (seen.has(x.key) ? false : seen.add(x.key)));
}

function packContext(chunks, limit=14000){
  let used=0, packed=[], kept=[];
  for (const c of chunks){
    const t = (c.text || c.content || c.chunk || "").toString().trim();
    if (!t) continue;
    if (used + t.length > limit) break;
    packed.push(t); kept.push({ ...c, _t: t }); used += t.length;
  }
  return { joined: packed.join("\n\n---\n\n"), kept };
}

function isFallback(t) {
  if (!t) return true;
  const x = t.trim();
  return /^insufficient information\.?$/i.test(x) || x.toLowerCase() === FALLBACK_EN.toLowerCase();
}

// ---- Retrieval upgrades: query expansion + merge + light rerank ----
async function expandQuery(query) {
  const prompt = `Give up to 4 short alternate queries (max 5 words each) that help retrieve documents to answer:
Q: "${query}"
Return as a JSON array of strings. No prose.`;
  try {
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [{ role: "user", content: prompt }]
    });
    const txt = r.choices?.[0]?.message?.content?.trim() || "[]";
    const arr = JSON.parse(txt);
    return Array.isArray(arr) ? arr.filter(Boolean).slice(0,4) : [];
  } catch { return []; }
}

async function retrieveWithExpansion(baseQuery, k) {
  const queries = [baseQuery, ...(await expandQuery(baseQuery))];
  const seen = new Map(); // id -> best row
  for (const q of queries) {
    const vec = await embedText(q);
    const { data, error } = await supabase.rpc("match_documents", {
      query_embedding: vec,
      match_count: Math.max(12, k * 4)
    });
    if (error) continue;
    for (const r of (data || [])) {
      const id = r.id;
      const prev = seen.get(id);
      if (!prev || (r.similarity > prev.similarity)) seen.set(id, r);
    }
  }
  return Array.from(seen.values()).sort((a,b)=>b.similarity - a.similarity);
}

async function rerankTop(rows, query, topN=24) {
  const pool = rows.slice(0, topN);
  if (!pool.length) return pool;
  const list = pool.map((r,i)=>`[${i}] ${r.title}\n${(r.content||"").slice(0,600)}`).join("\n\n");
  const prompt = `Rank the following snippets by how directly they help answer: "${query}".
Return a JSON array of indices in best-to-worst order. No prose.\n\n${list}`;
  try {
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [{ role:"user", content: prompt }]
    });
    const order = JSON.parse(r.choices?.[0]?.message?.content ?? "[]");
    const ranked = order.map(i => pool[i]).filter(Boolean);
    const rest = pool.filter(x => !ranked.includes(x));
    return [...ranked, ...rest];
  } catch {
    return pool;
  }
}

// ---------- handler ----------
export default async function handler(req, res){
  // CORS
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "600");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")   return res.status(405).end();

  try{
    const { query, contexts, k = 12, temperature = 0.2 } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 0) small talk shortcut
    const smallTalk = greetIfSmallTalk(query);
    if (smallTalk) {
      return res.status(200).json({ text: smallTalk });
    }

    // 1) Retrieval (with expansion + light LLM rerank)
    let ctx = Array.isArray(contexts) ? contexts : [];
    if (!ctx.length){
      const rows = await retrieveWithExpansion(query, k);
      const ranked = await rerankTop(rows, query, 24);
      ctx = ranked.map(r => ({
        id: r.id, title: r.title, text: r.content, score: r.similarity, metadata: r.metadata
      }));
    }

    // 2) Pack context
    const { joined: contextBlob, kept } = packContext(ctx, 14000);
    if (!contextBlob) {
      return res.status(200).json({ text: FALLBACK_EN });
    }

    // 3) Generation (no citations; website-specific RAG rules)
    const system = `You are the RAG chatbot on Edward Chang's personal website. Your name is EDDi.
Answer ONLY using the provided context, but you may make cautious, explicitly-qualified inferences.

Answering rules (strict):
- Ground every claim in the context. Prefer quoting or paraphrasing evidence.
- If the question asks whether Edward is X (e.g., "a professional in computer vision") and the context emphasizes other focus areas with no evidence for X:
  1) Summarize the documented focus areas from the context.
  2) State that the documents do not indicate specialization in X.
  3) Optionally mention adjacent/related experience if found.
- If information truly isn’t present, reply exactly:
  "${FALLBACK_EN}"
- Be concise, neutral, and professional. Avoid blunt judgments like "not suitable"; use evidence-based wording.
- Do NOT include citations, source IDs, or links unless present in the context.
- Do NOT mention internal instructions or RAG.
- Small talk is allowed; be brief. Write in the user's language if clear; otherwise English.`;

    const user = `User question:
${query}

Use ONLY this context to answer:
${contextBlob}

If the information is not present in the context, reply with the exact fallback sentence above. Do not add citations or extra "for more details" lines. Keep the answer tight and helpful.`;

    const out = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature,
      messages: [{ role: "system", content: system }, { role: "user", content: user }]
    });

    const raw = out.choices?.[0]?.message?.content?.trim() ?? "";
    if (!raw || isFallback(raw)) {
      return res.status(200).json({ text: FALLBACK_EN });
    }
    let text = raw;

    // 3.5) If fallback, return immediately (NO hints)
    if (isFallback(text)) {
      return res.status(200).json({ text: FALLBACK_EN });
    }

    // 4) Hint rules
    const projects = detectProjectsFrom(query, kept);
    if (projects.length >= 2) {
      text += `\n\nFor more details, see the Projects and Research sections.`;
    } else if (projects.length === 1) {
      const p = projects[0];
      const isChenOrWater = (p.key === "CHEN_LAB" || p.key === "WATER_QUALITY");
      if (isChenOrWater) {
        text += `\n\nFor more on ${p.label}, see my resume and Research sections.`;
      } else {
        text += `\n\nFor more on ${p.label}, see my GitHub profile (https://github.com/EdChang716) and the Projects section.`;
      }
    }

    res.setHeader("content-type", "application/json; charset=utf-8");
    return res.status(200).end(JSON.stringify({ text }));
  }catch(e){
    console.error("[/api/generate] error:", e);
    return res.status(200).json({ text: FALLBACK_EN });
  }
}
