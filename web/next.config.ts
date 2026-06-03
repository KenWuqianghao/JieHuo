import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  outputFileTracingRoot: path.join(__dirname),
  outputFileTracingExcludes: {
    "/search": [
      "./node_modules/onnxruntime-node/bin/napi-v3/darwin/**/*",
      "./node_modules/onnxruntime-node/bin/napi-v3/linux/arm64/**/*",
      "./node_modules/@img/sharp-darwin-arm64/**/*",
      "./node_modules/@img/sharp-libvips-darwin-arm64/**/*",
    ],
    "/api/route": [
      "./node_modules/onnxruntime-node/bin/napi-v3/darwin/**/*",
      "./node_modules/onnxruntime-node/bin/napi-v3/linux/arm64/**/*",
      "./node_modules/@img/sharp-darwin-arm64/**/*",
      "./node_modules/@img/sharp-libvips-darwin-arm64/**/*",
    ],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        sharp$: false,
        "onnxruntime-node$": false,
      };
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
