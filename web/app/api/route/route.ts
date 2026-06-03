import { NextRequest, NextResponse } from "next/server";
import { routeSearchQuery, sanitizeSearchQuery } from "@/lib/server-router";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  const rawQuery =
    request.nextUrl.searchParams.get("q") ?? request.nextUrl.searchParams.get("query");
  const query = sanitizeSearchQuery(rawQuery);

  if (!query) {
    return NextResponse.json({ error: "Missing required query parameter: q" }, { status: 400 });
  }

  try {
    const routed = await routeSearchQuery(query);
    return NextResponse.json(routed, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    console.error("JieHuo /api/route failed:", err);
    return NextResponse.json(
      { error: "Model routing unavailable", detail: err instanceof Error ? err.message : String(err) },
      { status: 503, headers: { "Cache-Control": "no-store" } }
    );
  }
}
