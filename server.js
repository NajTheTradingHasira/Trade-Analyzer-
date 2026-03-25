require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ──
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// ── Health Check ──
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    providers: {
      claude: !!process.env.ANTHROPIC_API_KEY,
      perplexity: !!process.env.PERPLEXITY_API_KEY,
    },
  });
});

// ── Claude Proxy ──
app.post("/api/claude", async (req, res) => {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return res.status(500).json({ error: "ANTHROPIC_API_KEY not configured on server" });

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: req.body.model || "claude-sonnet-4-20250514",
        max_tokens: req.body.max_tokens || 4000,
        system: req.body.system || "",
        messages: req.body.messages || [],
        tools: req.body.tools || [],
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data.error?.message || `Anthropic API error: ${response.status}`,
      });
    }

    res.json(data);
  } catch (err) {
    console.error("[Claude Proxy Error]", err.message);
    res.status(500).json({ error: `Proxy error: ${err.message}` });
  }
});

// ── Perplexity Proxy ──
app.post("/api/perplexity", async (req, res) => {
  // Allow client-side key OR server-side key
  const key = req.headers["x-pplx-key"] || process.env.PERPLEXITY_API_KEY;
  if (!key) return res.status(500).json({ error: "No Perplexity API key available. Set PERPLEXITY_API_KEY on server or pass your own key." });

  try {
    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: req.body.model || "sonar-pro",
        messages: req.body.messages || [],
        max_tokens: req.body.max_tokens || 4000,
        search_recency_filter: req.body.search_recency_filter || "week",
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data.error?.message || `Perplexity API error: ${response.status}`,
      });
    }

    res.json(data);
  } catch (err) {
    console.error("[Perplexity Proxy Error]", err.message);
    res.status(500).json({ error: `Proxy error: ${err.message}` });
  }
});

// ── Key Validation Endpoints ──
app.post("/api/perplexity/test", async (req, res) => {
  const key = req.headers["x-pplx-key"] || process.env.PERPLEXITY_API_KEY;
  if (!key) return res.status(400).json({ valid: false, error: "No key provided" });

  try {
    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: req.body.model || "sonar",
        messages: [{ role: "user", content: "Say OK" }],
        max_tokens: 10,
      }),
    });

    const data = await response.json();
    if (data.error) return res.json({ valid: false, error: data.error.message });
    res.json({ valid: true });
  } catch (err) {
    res.json({ valid: false, error: err.message });
  }
});

// ── Catch-all: serve frontend ──
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ── Start ──
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════╗
║   ⚡ AI TRADE ANALYZER SUITE                     ║
║   Server running on http://localhost:${PORT}        ║
║                                                  ║
║   Providers:                                     ║
║   Claude:     ${process.env.ANTHROPIC_API_KEY ? "✅ Key configured" : "❌ ANTHROPIC_API_KEY not set"}              ║
║   Perplexity: ${process.env.PERPLEXITY_API_KEY ? "✅ Key configured" : "⚠️  Client-side key mode"}              ║
╚══════════════════════════════════════════════════╝
  `);
});
