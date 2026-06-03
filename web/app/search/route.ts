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
    return NextResponse.redirect(new URL("/", request.url), {
      status: 302,
      headers: { "Cache-Control": "no-store" },
    });
  }

  const routed = await routeSearchQuery(query);
  return NextResponse.redirect(routed.targetUrl, {
    status: 302,
    headers: { "Cache-Control": "no-store" },
  });
}
