'use client';
import { useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { JoinedModelData } from '@/lib/data';

export function ModelList({ initialModels }: { initialModels: JoinedModelData[] }) {
  const [search, setSearch] = useState('');

  const filteredModels = initialModels.filter(m => {
    const term = search.toLowerCase();
    const name = (m.metadata?.name || m.yamlId).toLowerCase();
    const org = (m.metadata?.organization || '').toLowerCase();
    return name.includes(term) || org.includes(term);
  });

  return (
    <div className="space-y-6">
      <div className="max-w-sm">
        <Input 
          type="search" 
          placeholder="Search models or providers..." 
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredModels.map(model => (
          <Link key={model.id} href={`/models/${model.yamlId}`} className="group block h-full">
            <Card className="h-full flex flex-col transition-colors group-hover:border-primary/50 group-hover:bg-muted/50 cursor-pointer">
              <CardHeader>
                <CardTitle>{model.metadata?.name || model.yamlId}</CardTitle>
                <CardDescription>{model.metadata?.organization || 'Unknown Organization'}</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-end">
                <div className="flex flex-wrap gap-2 mt-4">
                  {model.metadata?.specs?.total_params && (
                    <Badge variant="secondary">{(model.metadata.specs.total_params / 1e9).toFixed(0)}B params</Badge>
                  )}
                  {model.metadata?.open_weights && (
                    <Badge variant="outline">Open Weights</Badge>
                  )}
                  <Badge variant="default" className="ml-auto">Score: {(model.prediction?.predicted_score * 100).toFixed(1)}</Badge>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
        {filteredModels.length === 0 && (
          <div className="col-span-full py-12 text-center text-muted-foreground">
            No models found matching "{search}"
          </div>
        )}
      </div>
    </div>
  );
}
