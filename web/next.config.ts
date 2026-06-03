import type { NextConfig } from "next";
import path from "path";

const transformersWeb = path.join(
  __dirname,
  "node_modules/@huggingface/transformers/dist/transformers.web.js"
);

const onnxNodeExclude = "./node_modules/onnxruntime-node/**/*";
const wasmRuntimeInclude = [
  "./node_modules/@huggingface/transformers/dist/transformers.web.js",
  "./node_modules/@huggingface/transformers/dist/*.wasm",
  "./node_modules/onnxruntime-web/dist/**/*",
  "./node_modules/onnxruntime-common/**/*",
];
const sharpExclude = [
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
  // Do not externalize @huggingface/transformers — Node "exports" would load transformers.node.mjs + onnxruntime-node.
  outputFileTracingIncludes: {
    "/search": wasmRuntimeInclude,
    "/api/route": wasmRuntimeInclude,
  },
  outputFileTracingExcludes: {
    "/search": [onnxNodeExclude, ...sharpExclude],
    "/api/route": [onnxNodeExclude, ...sharpExclude],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "@huggingface/transformers$": transformersWeb,
        "@huggingface/transformers/dist/transformers.node.mjs": transformersWeb,
        "@huggingface/transformers/dist/transformers.node.cjs": transformersWeb,
        "onnxruntime-node$": false,
        sharp$: false,
      };
    } else {
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
