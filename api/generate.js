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

// ---- Conversation helpers ----
function normalizeHistory(history = [], maxItems = 8, maxChars = 1200){
  const clean = (history || [])
    .filter(m => m && (m.role === "user" || m.role === "assistant") && m.content)
    .slice(-maxItems);
  let s = clean.map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`).join("\n");
  if (s.length > maxChars) s = s.slice(-maxChars); // 尾端保留
  return s;
}

// 用「對話歷史 + 原問題」改寫成可獨立檢索的查詢（英文短句）
async function rewriteQuery(question, history = []){
  const sys = `Rewrite the user's latest question into ONE short, standalone English search query.
- Resolve pronouns and references using chat history.
- Keep specific proper nouns (labs, projects, names).
- Output ONLY the query text, no quotes, no prose.`;
  const messages = [
    { role: "system", content: sys },
    { role: "user", content: JSON.stringify({ history, question }) }
  ];
  try{
    const out = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.1,
      messages
    });
    return (out.choices?.[0]?.message?.content || question).trim().replace(/^"|"$/g,"");
  }catch{
    return question;
  }
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
    const { query, contexts, k = 12, temperature = 0.2, history = [] } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 0) small talk shortcut
    const smallTalk = greetIfSmallTalk(query);
    if (smallTalk) {
      return res.status(200).json({ text: smallTalk });
    }

    // A) 以「最近對話」改寫查詢，避免代稱/指代
    const shortHistory = normalizeHistory(history, 8, 1200);
    const rewritten = await rewriteQuery(query, shortHistory);

    // B) Retrieval (rewritten -> expansion -> light rerank)
    let ctx = Array.isArray(contexts) ? contexts : [];
    if (!ctx.length){
      const rows = await retrieveWithExpansion(rewritten, k);
      const ranked = await rerankTop(rows, rewritten, 24);
      ctx = ranked.map(r => ({
        id: r.id, title: r.title, text: r.content, score: r.similarity, metadata: r.metadata
      }));
    }

    // C) Pack context
    const { joined: contextBlob, kept } = packContext(ctx, 14000);
    if (!contextBlob) {
      return res.status(200).json({ text: FALLBACK_EN });
    }

    // D) Generation（把對話歷史一併提供，只用來解讀語境；事實仍只能出自 context）
    const system = `You are the RAG chatbot on Edward Chang's personal website. Your name is EDDi.
Answer ONLY using the provided context; you may use the conversation to resolve references and tone.
If information truly isn’t present, reply exactly:
"${FALLBACK_EN}"
Be concise, neutral, and professional. No citations, source IDs, or hidden-policy talk.`;

    const convo = shortHistory || "(no prior messages)";
    const user = `Conversation so far:
${convo}

User question: ${query}

Use ONLY these knowledge-base snippets for factual claims:
${contextBlob}

If the answer is not in the snippets, use the exact fallback sentence above. Do not add extra "for more details" text unless instructed later. Keep it tight.`;

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

    // E) Hint rules（保留你的規則）
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
