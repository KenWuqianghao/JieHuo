import type { NextConfig } from "next";
import path from "path";

/** Keep Vercel serverless bundles under the 250 MB limit (see transformers.js #1164). */
const routeTraceIncludes = [
  "./node_modules/@huggingface/transformers/dist/transformers.web.js",
  "./node_modules/onnxruntime-web/**/*",
  "./node_modules/onnxruntime-common/**/*",
];

const routeTraceExcludes = [
  "./node_modules/onnxruntime-node/**/*",
  "./node_modules/onnxruntime-node/bin/**/*",
  "./node_modules/@img/sharp-darwin-arm64/**/*",
  "./node_modules/@img/sharp-darwin-x64/**/*",
  "./node_modules/@img/sharp-libvips-darwin-arm64/**/*",
  "./node_modules/@img/sharp-libvips-darwin-x64/**/*",
  "./node_modules/@img/sharp-linux-arm64/**/*",
  "./node_modules/@img/sharp-linuxmusl-arm64/**/*",
  "./node_modules/@img/sharp-win32-ia32/**/*",
  "./node_modules/@img/sharp-win32-x64/**/*",
  "./node_modules/sharp/**/*",
];

const nextConfig: NextConfig = {
  serverExternalPackages: ["onnxruntime-web", "onnxruntime-common"],
  outputFileTracingRoot: path.join(__dirname),
  outputFileTracingIncludes: {
    "/search": routeTraceIncludes,
    "/api/route": routeTraceIncludes,
  },
  outputFileTracingExcludes: {
    "/search": routeTraceExcludes,
    "/api/route": routeTraceExcludes,
  },
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };
    if (isServer) {
      config.resolve.alias["@huggingface/transformers$"] = path.join(
        __dirname,
        "node_modules/@huggingface/transformers/dist/transformers.web.js"
      );
    }
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };
    return config;
  },
  headers: async () => [
    {
      source: "/models/:path*",
      headers: [
        {
          key: "Cache-Control",
          value: "public, max-age=31536000, immutable",
        },
      ],
    },
  ],
};

export default nextConfig;
