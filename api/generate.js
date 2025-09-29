// api/generate.js
// - 9 intents with tailored response styles (no citations)
// - small talk allowed
// - BOOLEAN 無證據 → 溫和否定訊息 (非通用 fallback)
// - multi/single project "for more..." hints 保留
// - 支援 history + debug

import OpenAI from "openai";
import { classifyIntent, extractProjectHints } from "../lib/intent.js";
import { retrieveSmart, packContext } from "../lib/retrieve.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const ALLOW_ORIGIN = process.env.ALLOW_ORIGIN || "*";

const FALLBACK_EN =
  "We can’t answer your question based on the knowledge base. Please try a more specific query, or contact cc5375@columbia.edu for more information.";

const BOOL_FALLBACK =
  "Based on the knowledge base, there's no clear evidence for that. Please contact cc5375@columbia.edu for more information.";

/* ---------- utils ---------- */
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

function isFallback(t) {
  if (!t) return true;
  const x = t.trim();
  return /^insufficient information\.?$/i.test(x) || x.toLowerCase() === FALLBACK_EN.toLowerCase();
}

/* ---- 專案偵測（for more… 提示） ---- */
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

/* ---------- 9 種回應風格 (system prompt) ---------- */
function systemFor(intent){
  switch(intent){
    case "STAR": return `You are EDDi on Edward Chang's personal website.
Use ONLY the provided context. When the user asks about experiences, challenges, actions, impact, or results, write in STAR form as a single cohesive paragraph — do not use bullet points and do not include the labels "S:", "T:", "A:", or "R:".
Guidance: open with 1–2 sentences to establish the situation and task; flow into 1-2 sentences describing the key actions with concrete details (tools, models, data, evaluation); conclude with 1–2 sentences on the results, including metrics/awards/publications if present.
Keep it 4–6 sentences total, crisp and factual. No citations or source IDs. Do not speculate.`;

    case "FIT": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For fit/suitability:
- Summarize strengths, skills, and representative work that match the role.
- Tie each claim to evidence found in the context; if not evidenced, say "the documents do not indicate ...".
Be concise and professional, use 2-3 sentences in total. No citations.`;

    case "COMPARE": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For compare/contrast:
- Brief bullets for each entity (focus, methods, outcomes).
- Then 1–2 key differences, and optional complementary aspects.
Neutral tone. No citations.`;

    case "BOOLEAN": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For yes/no questions:
- If explicit evidence exists, answer yes/no and reference that evidence in wording (no formal citations).
- If evidence is unclear or absent, reply: "${BOOL_FALLBACK}"
Be brief and avoid speculation, use 2-4 sentences in total.`;

    case "LIST": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. When asked to list (skills, projects, publications, awards):
- Return a tight bullet list (3–8 items).
- Prefer specific names (project titles, courses, tools, venues) found in the context.
No citations.`;

    case "TIMELINE": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For timeline/when questions:
- Provide a chronological sequence with dates or ranges if available.
- If some dates are missing, state what's known and what's not.
No citations.`;

    case "SUMMARY": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For summary/overview/bio:
- Write 3–5 concise sentences highlighting education, focus areas, representative work, and recognitions.
No citations.`;

    case "PROJECT": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For a single project deep-dive:
- Give a 3–6 sentence overview: goal, dataset/scope, methods/models, and outcomes/impact, using STAR method.
- Mention metrics/awards if present.
No citations.`;

    case "CONTACT": return `You are EDDi on Edward Chang's personal website.
Use ONLY the context. For contact/logistics:
- Provide email, GitHub, LinkedIn, location, availability if present in the documents.
- Keep it compact. No citations.`;

    default: // FACTUAL
      return `You are EDDi on Edward Chang's personal website.
Answer ONLY using the provided context. If information is missing, reply exactly:
"${FALLBACK_EN}"
Be concise, neutral, and professional. Do NOT include citations or source IDs.`;
  }
}

/* ---------- handler ---------- */
export default async function handler(req, res){
  // CORS
  res.setHeader("Access-Control-Allow-Origin", ALLOW_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "600");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")   return res.status(405).end();

  try{
    const { query, k = 12, temperature = 0.2, history = [], debug = false } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 0) small talk
    const smallTalk = greetIfSmallTalk(query);
    if (smallTalk) return res.status(200).json({ text: smallTalk });

    // 1) 意圖 & 專案線索
    const intent = await classifyIntent(query);
    const project_hints = extractProjectHints(query);

    // 2) 智慧檢索（對話改寫 + HyDE + 展開 + 重排）
    const rows = await retrieveSmart(query, { k: Math.max(18, k*2), history, project_hints });
    const { joined: contextBlob, kept } = packContext(rows, 14000);

    // 3) 生成
    const sys = systemFor(intent);
    const convo = (history || [])
      .slice(-8)
      .map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n") || "(no prior messages)";

    const userMsg = `Conversation so far:
${convo}

User question: ${query}

Use ONLY these knowledge-base snippets for factual claims:
${contextBlob || "(no snippets)"} 

Follow the intent-specific guidance. Keep it crisp.`;

    const out = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature,
      max_tokens: tokenCap
      messages: [{ role: "system", content: sys }, { role: "user", content: userMsg }]
    });

    let text = out.choices?.[0]?.message?.content?.trim() ?? "";

    // 4) 缺證據時：BOOLEAN 走溫和否定；其它走通用 fallback
    if (!text || isFallback(text)){
      text = (intent === "BOOLEAN") ? BOOL_FALLBACK : FALLBACK_EN;
    }

    // 5) for-more 提示（依你規則）
    if (contextBlob){
      const projects = detectProjectsFrom(query, kept);
      if (projects.length > 3) {
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
    }

    const payload = debug
      ? { text, intent, retrieved: rows.length, sample: rows.slice(0,5).map(r => ({id:r.id,title:r.title})) }
      : { text };
    const INTENT_SENTENCE_CAP = {
      STAR: 7,          // 以段落敘事式 STAR：4–7 句
      FIT: 5,           // 適配/職涯：3–5 句
      COMPARE: 6,       // 比較：4–6 句
      FACTUAL: 3,       // 事實型：1–3 句
      SMALL_TALK: 2,    // 寒暄：1–2 句
      GENERIC: 4
    };

    // 依意圖給 token 上限（硬上限，避免失控）
    // 英文粗估： words ≈ 0.75 * tokens；中文建議給寬一些
    const INTENT_TOKEN_CAP = {
      STAR: 260,
      FIT: 180,
      COMPARE: 220,
      FACTUAL: 120,
      SMALL_TALK: 80,
      GENERIC: 160
    };

    // 全域保險（若沒命中意圖就用這個）
    const MAX_TOKENS_GLOBAL = Number(process.env.MAX_TOKENS_GLOBAL || 200);

    res.setHeader("content-type", "application/json; charset=utf-8");
    return res.status(200).end(JSON.stringify(payload));
  }catch(e){
    console.error("[/api/generate] error:", e);
    return res.status(200).json({ text: FALLBACK_EN });
  }
}
