'use client';
import { useState } from 'react';
import { Input } from '@/components/ui/input';

export function BenchmarkList({ initialTasks }: { initialTasks: { name: string, sampleCount: number }[] }) {
  const [search, setSearch] = useState('');

  const filteredTasks = initialTasks.filter(t => 
    t.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-4">
      <div className="max-w-xs">
        <Input 
          type="search" 
          placeholder="Search source tasks..." 
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-8 text-sm shadow-none rounded-sm bg-gray-50 border-gray-200"
        />
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="py-2 text-left font-medium text-gray-500">Task Name</th>
            <th className="py-2 text-right font-medium text-gray-500">Labeled Samples</th>
          </tr>
        </thead>
        <tbody>
          {filteredTasks.map(task => (
            <tr key={task.name} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
              <td className="py-2.5 font-medium text-gray-900">{task.name}</td>
              <td className="py-2.5 text-right font-mono text-xs text-gray-600">{task.sampleCount.toLocaleString()}</td>
            </tr>
          ))}
          {filteredTasks.length === 0 && (
            <tr>
              <td colSpan={2} className="py-8 text-center text-gray-500">
                No tasks found matching "{search}"
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
