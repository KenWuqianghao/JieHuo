import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY ?? process.env.SUPABASE_ANON_KEY;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, predicted_label, confidence, correct } = body;

    if (!query || !predicted_label) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    const record = {
      query: String(query).slice(0, 500),
      predicted_label: String(predicted_label),
      confidence: Number(confidence) || 0,
      correct: Boolean(correct),
      created_at: new Date().toISOString(),
    };

    if (supabaseUrl && supabaseKey) {
      const supabase = createClient(supabaseUrl, supabaseKey);
      const { error } = await supabase.from("query_feedback").insert(record);
      if (error) {
        console.error("Supabase insert error:", error.message);
        return NextResponse.json({ ok: false, stored: false, error: error.message });
      }
      return NextResponse.json({ ok: true, stored: true });
    }

    // Fallback: log locally when Supabase is not configured
    console.log("[feedback]", JSON.stringify(record));
    return NextResponse.json({ ok: true, stored: false, message: "Logged locally (Supabase not configured)" });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Unknown error" },
      { status: 500 }
    );
  }
}
