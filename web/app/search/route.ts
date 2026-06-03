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
  const mode: RoutingMode =
    request.nextUrl.searchParams.get("mode") === "model" ? "model" : "heuristic";

  if (!query) {
    return NextResponse.redirect(new URL("/", request.url), {
      status: 302,
      headers: { "Cache-Control": "no-store" },
    });
  }

  const routed = await routeSearchQuery(query, mode);
  return NextResponse.redirect(routed.targetUrl, {
    status: 302,
    headers: { "Cache-Control": "no-store" },
  });
}
