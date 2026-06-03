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
    return NextResponse.redirect(new URL("/", request.url), {
      status: 302,
      headers: { "Cache-Control": "no-store" },
    });
  }

  try {
    const routed = await routeSearchQuery(query);
    return NextResponse.redirect(routed.targetUrl, {
      status: 302,
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    console.error("JieHuo /search route failed:", err);
    return NextResponse.json(
      { error: "Model routing unavailable", detail: err instanceof Error ? err.message : String(err) },
      { status: 503, headers: { "Cache-Control": "no-store" } }
    );
  }
}
