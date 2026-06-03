import { NextRequest, NextResponse } from "next/server";
import type { RoutingMode } from "@/lib/server-router";
import { routeSearchQuery, sanitizeSearchQuery } from "@/lib/server-router";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  const rawQuery =
    request.nextUrl.searchParams.get("q") ?? request.nextUrl.searchParams.get("query");
  const query = sanitizeSearchQuery(rawQuery);
  const modeParam = request.nextUrl.searchParams.get("mode");
  const mode: RoutingMode = modeParam === "model" ? "model" : "heuristic";

  if (!query) {
    return NextResponse.json({ error: "Missing required query parameter: q" }, { status: 400 });
  }

  const routed = await routeSearchQuery(query, mode);
  return NextResponse.json(routed, {
    headers: { "Cache-Control": "no-store" },
  });
}
