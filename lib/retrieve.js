// lib/retrieve.js
// 取用：對話改寫 → HyDE → 查詢展開 → 多查詢合併去重 → 關鍵字加權 → LLM 輕重排 → 打包

import OpenAI from "openai";
import { embedText } from "./embeddings.js";
import { supabase } from "./supabase.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* -------- 打包 -------- */
export function packContext(chunks, limit=14000){
  let used=0, packed=[], kept=[];
  for (const c of chunks||[]){
    const t = (c.content || c.text || c.chunk || "").toString().trim();
    if (!t) continue;
    if (used + t.length > limit) break;
    packed.push(t); kept.push({ ...c, _t: t });
    used += t.length;
  }
  return { joined: packed.join("\n\n---\n\n"), kept };
}

/* -------- 對話改寫 -------- */
async function rewriteQuery(question, history = []){
  const sys = `Rewrite the user's latest question into ONE short, standalone English search query.
- Resolve pronouns using chat history.
- Keep proper nouns (labs, projects, names).
- Output ONLY the query text.`;
  try{
    const out = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.1,
      messages: [
        { role:"system", content: sys },
        { role:"user", content: JSON.stringify({ history, question }) }
      ]
    });
    return (out.choices?.[0]?.message?.content || question).trim().replace(/^"|"$/g,"");
  }catch{ return question; }
}

/* -------- HyDE（假想答案嵌入檢索） -------- */
async function hydeDraft(query){
  const sys = `Write one short, neutral paragraph that could plausibly answer the user's question about Edward Chang's background/projects.
Do NOT include proper nouns unless they appear generic. 80–120 words.`;
  try{
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.6,
      messages: [{ role:"system", content: sys }, { role:"user", content: query }]
    });
    return (r.choices?.[0]?.message?.content || "").slice(0, 900);
  }catch{ return ""; }
}

/* -------- 查詢展開 -------- */
async function expandQuery(query) {
  const prompt = `Give up to 4 short alternate queries (max 5 words each) that help retrieve documents to answer:
Q: "${query}"
Return as a JSON array of strings.`;
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

/* -------- 檢索核心 -------- */
function keywordBoostScore(text, hints){
  if (!hints?.length || !text) return 0;
  const bag = text.toLowerCase();
  let s = 0;
  for (const h of hints){
    if (bag.includes(h.toLowerCase())) s += 0.5; // 簡單加權
  }
  return Math.min(s, 1.5);
}

export async function retrieveSmart(query, { k=24, history=[], project_hints=[] } = {}){
  // 1) 改寫
  const shortHistory = (history||[])
    .filter(m => m && (m.role==="user" || m.role==="assistant") && m.content)
    .slice(-8)
    .map(m => `${m.role==="user"?"User":"Assistant"}: ${m.content}`)
    .join("\n");
  const rewritten = await rewriteQuery(query, shortHistory);

  // 2) HyDE
  const draft = await hydeDraft(rewritten);

  // 3) 展開
  const expanded = await expandQuery(rewritten);

  // 4) 多查詢合併
  const queries = [rewritten, ...expanded];
  if (draft) queries.push(draft);

  const seen = new Map(); // id -> best row
  for (const q of queries){
    const vec = await embedText(q);
    const { data } = await supabase.rpc("match_documents", {
      query_embedding: vec,
      match_count: Math.max(12, k)
    });
    for (const r of (data || [])) {
      const text = r.content || "";
      const kb = keywordBoostScore(text, project_hints);
      const score = (r.similarity || 0) + 0.15*kb; // 混合分數
      const prev = seen.get(r.id);
      if (!prev || score > prev._score) {
        seen.set(r.id, { ...r, _score: score });
      }
    }
  }

  let rows = Array.from(seen.values()).sort((a,b)=>b._score - a._score);

  // 5) LLM 輕量重排
  const pool = rows.slice(0, Math.max(18, k));
  if (pool.length){
    const list = pool.map((r,i)=>`[${i}] ${r.title}\n${(r.content||"").slice(0,600)}`).join("\n\n");
    const prompt = `Rank the following snippets by how directly they help answer: "${query}".
Return a JSON array of indices in best-to-worst order. No prose.\n\n${list}`;
    try{
      const rr = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0,
        messages: [{ role:"user", content: prompt }]
      });
      const order = JSON.parse(rr.choices?.[0]?.message?.content ?? "[]");
      const ranked = order.map(i => pool[i]).filter(Boolean);
      const rest = pool.filter(x => !ranked.includes(x));
      rows = [...ranked, ...rest, ...rows.slice(pool.length)];
    }catch{}
  }

  // 6) 轉為 generate 消費格式
  return rows.map(r => ({
    id: r.id,
    title: r.title,
    text: r.content,
    score: r._score ?? r.similarity,
    metadata: r.metadata
  }));
}
