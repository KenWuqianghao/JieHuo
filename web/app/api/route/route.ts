import { NextRequest, NextResponse } from "next/server";
import { routeSearchQuery, sanitizeSearchQuery } from "@/lib/server-router";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 10;

export async function GET(request: NextRequest) {
  const rawQuery =
    request.nextUrl.searchParams.get("q") ?? request.nextUrl.searchParams.get("query");
  const query = sanitizeSearchQuery(rawQuery);

  if (!query) {
    return NextResponse.json({ error: "Missing required query parameter: q" }, { status: 400 });
  }

  const routed = await routeSearchQuery(query);
  return NextResponse.json(routed, {
    headers: { "Cache-Control": "no-store" },
  });
}
