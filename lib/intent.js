// lib/intent.js
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * classifyIntent(query) -> { intent, confidence, project_hints[] }
 * intent ∈ STAR | FIT | FACTUAL | SMALL_TALK
 * project_hints ∈ ["CHEN_LAB","WATER_QUALITY","ANNA","CROSS_DOMAIN","ARG_MINING","CAYIN"]
 */
export async function classifyIntent(query) {
  const prompt = `
You are a classifier. Classify the user's question into one of:
- STAR: asks for stories, experience, challenges, impact, results (e.g., "tell me about a time...", "how did he handle...", "what was the result?")
- FIT: asks about suitability/strengths for roles, skills fit, recruiter-like questions.
- FACTUAL: specific facts (degree, dates, titles, where, when, metrics).
- SMALL_TALK: greetings or chit-chat without info intent.

Also provide related project hints if the question points to any of these keys:
["CHEN_LAB","WATER_QUALITY","ANNA","CROSS_DOMAIN","ARG_MINING","CAYIN"]

Keyword guidance (soft):
- CHEN_LAB: "CHEN Lab", "neuroscience", "Pin-Hao Chen", "conversation", "dyadic", "nonverbal"
- WATER_QUALITY: "water quality", "LSTM-ED", "Taipei Bridge", "imputation", "MICE-RF"
- ANNA: "Anna Karenina", "well-being", "friendship representation"
- CROSS_DOMAIN: "cross-domain", "semantic similarity", "embedding bias"
- ARG_MINING: "argument mining"
- CAYIN: "CAYIN", "generative ad", "internship"

Return ONLY compact JSON:
{"intent":"STAR|FIT|FACTUAL|SMALL_TALK","confidence":0.0-1.0,"project_hints":["..."]}

User question: "${query}"
`;

  try {
    const r = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0,
      messages: [{ role: "user", content: prompt }]
    });
    const txt = r.choices?.[0]?.message?.content?.trim() ?? "";
    const parsed = JSON.parse(txt);
    const intent = ["STAR","FIT","FACTUAL","SMALL_TALK"].includes(parsed.intent) ? parsed.intent : "FACTUAL";
    const hints = Array.isArray(parsed.project_hints) ? parsed.project_hints.filter(x =>
      ["CHEN_LAB","WATER_QUALITY","ANNA","CROSS_DOMAIN","ARG_MINING","CAYIN"].includes(x)
    ) : [];
    const confidence = Math.max(0, Math.min(1, Number(parsed.confidence ?? 0.6)));
    return { intent, confidence, project_hints: hints };
  } catch {
    return { intent: "FACTUAL", confidence: 0.5, project_hints: [] };
  }
}
