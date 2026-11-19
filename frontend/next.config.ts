import type { NextConfig } from "next";

// Set by the GitHub Pages workflow: "/compl-ai" on <org>.github.io, empty on a custom domain.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
