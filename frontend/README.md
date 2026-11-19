This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy (GitHub Pages)

Pushes to `main` or `frontend` that touch `frontend/` run `.github/workflows/deploy-frontend.yml`, which statically exports the site and publishes it to https://compl-ai.github.io/compl-ai/ (or the custom domain, once configured in Settings → Pages).

The workflow sets `NEXT_PUBLIC_BASE_PATH` (`/compl-ai` on github.io, empty on a custom domain). `next/link` applies it automatically; wrap other `/public` URLs (plain `<a>`, `<img>`, `fetch`) in `withBasePath()` from `@/lib/utils`. To test locally: `NEXT_PUBLIC_BASE_PATH=/compl-ai npm run build`.
