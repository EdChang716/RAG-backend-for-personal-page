// api/generate.js
// Rules:
// - no citations
// - small talk allowed
// - fallback in EN with contact
// - hints: multi-project => generic "for more..."
//          single-project => CHEN/WaterQuality -> resume+Research
//                            others            -> GitHub+Projects

import OpenAI from "openai";
import { classifyIntent } from "../lib/intent.js";
import { retrieveSmart, packContext } from "../lib/retrieve.js";

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

// ---- Project detectors (for hint lines) ----
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

function isFallback(t) {
  if (!t) return true;
  const x = t.trim();
  return /^insufficient information\.?$/i.test(x) || x.toLowerCase() === FALLBACK_EN.toLowerCase();
}

// ---- Templates ----
function sysSTAR(fallback){
  return `You are the RAG chatbot on Edward Chang's personal website. Your name is EDDi.
Answer ONLY using the provided context. If missing, reply exactly:
"${fallback}"

When the user asks about experiences/challenges/impact/results, respond in STAR format:
- Situation: 1–2 lines summarizing the problem/opportunity and setting.
- Task: 1 line describing the specific responsibility or goal.
- Action: 2–4 bullets focusing on concrete technical steps (tools, models, data, evaluation).
- Result: 1–2 lines with measurable outcomes (metrics, awards, impact) if present in context.

Be crisp, factual, and avoid speculation. Do NOT include citations or source IDs.`;
}

function sysFIT(fallback){
  return `You are the RAG chatbot on Edward Chang's personal website. Your name is EDDi.
Answer ONLY using the provided context. If missing, reply exactly:
"${fallback}"

For fit/suitability questions:
- Summarize documented strengths, skills, and representative projects relevant to the role.
- Use evidence-based wording from the context; avoid blunt negatives.
- If something isn't evidenced, say "the documents do not indicate ...".
Be concise and professional. No citations.`;
}

function sysFACTUAL(fallback){
  return `You are the RAG chatbot on Edward Chang's personal website. Your name is EDDi.
Answer ONLY using the provided context. If information is missing, reply exactly:
"${fallback}"
Answer concisely with verified facts. No citations.`;
}

function sysSMALL_TALK(){
  return `You are EDDi on Edward's site. Be brief and friendly. If the user asks for info, you can help with background, labs, projects, resume, or GitHub. Do not invent facts.`;
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
    const { query, k = 12, temperature = 0.2, sessionId = null } = await readJson(req);
    if (!query) return res.status(400).json({ error: "missing query" });

    // 0) small talk shortcut
    const smallTalk = greetIfSmallTalk(query);
    if (smallTalk) {
      return res.status(200).json({ text: smallTalk });
    }

    // 1) 意圖分類
    const { intent, project_hints } = await classifyIntent(query);

    // 2) （選配）會話摘要（這裡先不強制，需要你之後自己加表，沒有就當空字串）
    const historySnippet = ""; // 留白：未建 conversations 表時不影響功能

    // 3) 智慧檢索
    const rows = await retrieveSmart(query, { k: Math.max(18, k*2), project_hints, historySnippet });
    const { joined: contextBlob, kept } = packContext(rows, 14000);
    if (!contextBlob) {
      return res.status(200).json({ text: FALLBACK_EN });
    }

    // 4) 選模板 & 生成
    let sys;
    switch (intent) {
      case "STAR":       sys = sysSTAR(FALLBACK_EN); break;
      case "FIT":        sys = sysFIT(FALLBACK_EN); break;
      case "SMALL_TALK": sys = sysSMALL_TALK(); break;
      default:           sys = sysFACTUAL(FALLBACK_EN); break;
    }

    const userMsg = `User question:
${query}

Use ONLY this context to answer:
${contextBlob}

If the information is not present, reply with the exact fallback sentence (no extras). keep it concise.`;

    const out = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature,
      messages: [{ role: "system", content: sys }, { role: "user", content: userMsg }]
    });

    const raw = out.choices?.[0]?.message?.content?.trim() ?? "";
    if (!raw || isFallback(raw)) {
      return res.status(200).json({ text: FALLBACK_EN });
    }
    let text = raw;

    // 5) 你的提示規則（多/單專案的「for more...」）
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
