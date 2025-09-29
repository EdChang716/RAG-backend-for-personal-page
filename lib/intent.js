// lib/intent.js
// 9 類意圖：STAR / FIT / COMPARE / BOOLEAN / LIST / TIMELINE / SUMMARY / PROJECT / CONTACT (+ FACTUAL fallback)

import OpenAI from "openai";
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 專案關鍵詞（供 PROJECT 類型與檢索加權）
const PROJECT_HINTS = [
  { key: "CHEN_LAB", re: /(chen\s+lab|computational\s+human|neuroscience\s+lab|pin-?hao|p[-\.]?\s*h\s*a?\.?\s*chen)/i },
  { key: "WATER_QUALITY", re: /(water\s+quality|lstm[-\s]?ed|taipei\s+bridge|mice[-\s]?rf|imputation)/i },
  { key: "ANNA", re: /(anna\s*karenina|well[-\s]?being|friendship)/i },
  { key: "CROSS_DOMAIN", re: /(cross[-\s]?domain|semantic\s+similarity|embedding[s]?\s+bias)/i },
  { key: "ARG_MINING", re: /(argument\s+mining|earnings\s+call|premise|claim)/i },
  { key: "CAYIN", re: /(cayin)/i },
];

export function extractProjectHints(q){
  const bag = q || "";
  return PROJECT_HINTS.filter(h => h.re.test(bag)).map(h => h.key);
}

export async function classifyIntent(query){
  const q = (query||"").trim();

  // 1) 快速規則（省 token）
  if (/\b(compare|contrast|difference|vs\.?|versus|diff between)\b/i.test(q)) return "COMPARE";
  if (/^\s*(is|are|does|do|did|has|have|can|could|will|was|were)\b/i.test(q)) return "BOOLEAN";
  if (/\b(when|timeline|since|from\s+\d{4}|until|expected|graduate|start(ed)?|end(ed)?)\b/i.test(q)) return "TIMELINE";
  if (/\b(list|which|what are (his|her) (skills|projects|papers|publications|awards))\b/i.test(q)) return "LIST";
  if (/\b(summarize|overview|background|bio|who is he)\b/i.test(q)) return "SUMMARY";
  if (/\b(tell me more about|deep[-\s]?dive|details of|explain)\b/i.test(q) || extractProjectHints(q).length) return "PROJECT";
  if (/\b(contact|email|reach|linkedin|github|location|where is he)\b/i.test(q)) return "CONTACT";
  if (/\b(experience|challenge|impact|result|accomplish|led|built|implemented|tell me about a time)\b/i.test(q)) return "STAR";
  if (/\b(fit|suitable|good (candidate|fit)|strengths?|weakness|why hire|why him)\b/i.test(q)) return "FIT";

  // 2) LLM 補判（避免漏網）
  const sys = `Classify the user's question into exactly one of:
- STAR
- FIT
- COMPARE
- BOOLEAN
- LIST
- TIMELINE
- SUMMARY
- PROJECT
- CONTACT
- FACTUAL  (fallback)

Return ONLY the label (uppercase).`;
  try{
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [{ role:"system", content: sys }, { role:"user", content: q }]
    });
    const lab = (r.choices?.[0]?.message?.content || "").trim().toUpperCase();
    if (["STAR","FIT","COMPARE","BOOLEAN","LIST","TIMELINE","SUMMARY","PROJECT","CONTACT","FACTUAL"].includes(lab)) return lab;
  }catch{}
  return "FACTUAL";
}
