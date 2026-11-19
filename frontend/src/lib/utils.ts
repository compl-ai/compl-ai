export { cn } from "cn"

// Prefix a /public asset URL with the deploy base path; next/link handles this itself, plain <a>/<img> don't.
export const withBasePath = (p: string) => `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}${p}`
