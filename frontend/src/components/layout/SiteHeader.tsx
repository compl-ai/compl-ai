import Link from 'next/link';
import { Hexagon, Search } from 'lucide-react';

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 w-full bg-transparent">
      <div className="container flex h-16 items-center px-4 md:px-8 max-w-[1400px] mx-auto justify-between">
        <div className="flex items-center">
          <Link href="/" className="mr-8 flex items-center space-x-2">
            <Hexagon className="w-6 h-6 text-primary fill-transparent stroke-2" />
            <span className="font-bold text-lg tracking-tight">
              COMPL-AI
            </span>
          </Link>
          <nav className="flex items-center space-x-8 text-sm font-semibold text-gray-500">
            <Link href="/" className="transition-colors hover:text-gray-900 flex items-center">
              Index <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 ml-1 mb-2"></span>
            </Link>
            <Link href="/models" className="transition-colors hover:text-gray-900">Models</Link>
            <Link href="/benchmarks" className="transition-colors hover:text-gray-900">Benchmarks</Link>
            <Link href="/methodology" className="transition-colors hover:text-gray-900">Methodology</Link>
          </nav>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="relative hidden md:flex items-center border border-gray-200 bg-white rounded-full px-3 py-1.5 w-64 shadow-sm">
            <Search className="w-4 h-4 text-gray-400 mr-2" />
            <input type="text" placeholder="Search (#K)" className="bg-transparent border-none outline-none text-sm w-full placeholder:text-gray-400" />
          </div>
          <div className="hidden lg:flex items-center bg-gray-900 text-white px-3 py-1.5 rounded-md font-mono text-xs shadow-sm">
            <span className="text-gray-400 mr-2">{'>'}</span> 
            <span>complai <span className="text-indigo-300">index</span> mymodel</span>
          </div>
          <a href="https://github.com/compl-ai/compl-ai" target="_blank" rel="noreferrer" className="text-gray-500 hover:text-gray-900 transition-colors">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.24c3-.34 6-1.5 6-6.76 0-1.5-.5-2.8-1.4-3.8.1-.3.6-1.8-.1-3.8 0 0-1.2-.4-3.9 1.4a12.3 12.3 0 0 0-7 0C5.6 2.6 4.4 3 4.4 3c-.7 2-.2 3.5-.1 3.8A6.5 6.5 0 0 0 3 10.6c0 5.2 3 6.4 6 6.76-.8.8-1 2-1 3.24v4"></path>
              <path d="M4 19c-2 1-4 0-4-0"></path>
            </svg>
          </a>
        </div>
      </div>
    </header>
  );
}
