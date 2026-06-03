import type { NextConfig } from "next";
import path from "path";

const onnxPlatformExcludes = [
  "./node_modules/onnxruntime-node/bin/napi-v3/darwin/**/*",
  "./node_modules/onnxruntime-node/bin/napi-v3/win32/**/*",
  "./node_modules/onnxruntime-node/bin/napi-v3/linux/arm64/**/*",
  "./node_modules/onnxruntime-node/bin/napi-v3/linux/arm/**/*",
  "./node_modules/@img/sharp-darwin-arm64/**/*",
  "./node_modules/@img/sharp-darwin-x64/**/*",
  "./node_modules/@img/sharp-libvips-darwin-arm64/**/*",
  "./node_modules/@img/sharp-libvips-darwin-x64/**/*",
  "./node_modules/@img/sharp-linux-arm64/**/*",
  "./node_modules/@img/sharp-linuxmusl-arm64/**/*",
  "./node_modules/@img/sharp-win32-ia32/**/*",
  "./node_modules/@img/sharp-win32-x64/**/*",
];

const nextConfig: NextConfig = {
  outputFileTracingRoot: path.join(__dirname),
  outputFileTracingExcludes: {
    "/search": onnxPlatformExcludes,
    "/api/route": onnxPlatformExcludes,
  },
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
    };
    if (!isServer) {
      config.resolve.alias["onnxruntime-node$"] = false;
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
