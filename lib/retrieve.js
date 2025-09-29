// lib/retrieve.js
import OpenAI from "openai";
import { embedText } from "./embeddings.js";
import { supabase } from "./supabase.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function uniqById(rows) {
  const m = new Map();
  for (const r of rows) {
    const prev = m.get(r.id);
    if (!prev || r.similarity > prev.similarity) m.set(r.id, r);
  }
  return Array.from(m.values());
}

function similarByOverlap(a, b) {
  // 粗暴去重：看文字重疊比例，避免大量相鄰 chunk 重複
  if (!a || !b) return 0;
  const A = a.replace(/\s+/g, " ").toLowerCase();
  const B = b.replace(/\s+/g, " ").toLowerCase();
  const min = Math.min(A.length, B.length);
  if (min < 400) return 0;
  const needle = A.slice(0, Math.min(1200, A.length));
  return B.includes(needle) ? 1 : 0;
}

function diversify(rows, maxN = 24) {
  const picked = [];
  for (const r of rows) {
    const tooSimilar = picked.some(p => similarByOverlap(p.content || p.text || "", r.content || r.text || ""));
    if (!tooSimilar) picked.push(r);
    if (picked.length >= maxN) break;
  }
  return picked;
}

async function expandQuery(base) {
  const prompt = `Give up to 4 short alternate queries (max 5 words each) that help retrieve documents to answer:
Q: "${base}"
Return as a JSON array of strings. No prose.`;
  try {
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [{ role: "user", content: prompt }]
    });
    const arr = JSON.parse(r.choices?.[0]?.message?.content ?? "[]");
    return Array.isArray(arr) ? arr.filter(Boolean).slice(0, 4) : [];
  } catch {
    return [];
  }
}

function hintLabel(key) {
  switch (key) {
    case "CHEN_LAB": return "CHEN Lab";
    case "WATER_QUALITY": return "Water Quality Prediction";
    case "ANNA": return "Anna Karenina / Well-being";
    case "CROSS_DOMAIN": return "Cross-domain Text Semantic Similarity";
    case "ARG_MINING": return "Argument Mining";
    case "CAYIN": return "CAYIN Generative AI";
    default: return key;
  }
}

async function searchOnce(query, k) {
  const vec = await embedText(query);
  const { data, error } = await supabase.rpc("match_documents", {
    query_embedding: vec,
    match_count: Math.max(12, k)
  });
  if (error) throw error;
  return (data || []).map(r => ({
    id: r.id,
    title: r.title,
    content: r.content,
    similarity: r.similarity,
    metadata: r.metadata
  }));
}

/**
 * retrieveSmart(query, {k, project_hints, historySnippet})
 * - 先全域擴搜
 * - 再對每個 hint 做「偏置搜尋」（把 hint label 附加到查詢字串）
 * - 合併、去重、多樣化
 * - 覆蓋度不足 → 再放寬一次
 */
export async function retrieveSmart(query, { k = 24, project_hints = [], historySnippet = "" } = {}) {
  const base = historySnippet ? `${query}\n\nRecent context: ${historySnippet}` : query;

  // 1) 全域擴搜（base + expansions）
  const exps = await expandQuery(base);
  let pool = [];
  const globalQs = [base, ...exps];
  for (const q of globalQs) {
    try {
      const rows = await searchOnce(q, Math.max(24, k));
      pool.push(...rows);
    } catch {}
  }

  // 2) 專案偏置（軟性：把 label 拼進 query 再搜）
  for (const key of project_hints) {
    const q2 = `${base}\n\nFocus: ${hintLabel(key)}`;
    try {
      const rows = await searchOnce(q2, Math.max(24, k));
      pool.push(...rows);
    } catch {}
  }

  // 3) 合併 & 初步排序
  pool = uniqById(pool).sort((a, b) => b.similarity - a.similarity);

  // 4) 多樣化
  let picked = diversify(pool, Math.max(18, Math.floor(k * 1.2)));

  // 5) 覆蓋度檢查，不足就放寬一次
  const totalChars = picked.reduce((s, r) => s + (r.content?.length || 0), 0);
  const distinctDocs = new Set(picked.map(r => r.metadata?.doc_id || r.metadata?.path || r.id.split(":")[0])).size;

  if (totalChars < 1200 || distinctDocs < 3) {
    // 再擴一次（大 k）
    const bigger = [];
    for (const q of [base, ...exps]) {
      try {
        const rows = await searchOnce(q, 48);
        bigger.push(...rows);
      } catch {}
    }
    pool = uniqById([...pool, ...bigger]).sort((a, b) => b.similarity - a.similarity);
    picked = diversify(pool, 24);
  }

  return picked;
}

/** 把多段 snippet 接成一段上下文，限制總字數 */
export function packContext(chunks, limit = 14000) {
  let used = 0;
  const packed = [];
  const kept = [];
  for (const c of chunks) {
    const t = (c.content || c.text || "").toString().trim();
    if (!t) continue;
    if (used + t.length > limit) break;
    packed.push(t);
    kept.push({ ...c, _t: t });
    used += t.length;
  }
  return { joined: packed.join("\n\n---\n\n"), kept };
}
