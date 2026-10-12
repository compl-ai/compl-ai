'use client';

export function IndexBarChart({ data }: { data: { label: string, score: number, globalAvg: number }[] }) {
  return (
    <div className="space-y-4">
      {data.map(item => (
        <div key={item.label} className="grid grid-cols-12 gap-4 items-center">
          <div className="col-span-4 font-medium text-sm text-gray-900 truncate">
            {item.label}
          </div>
          <div className="col-span-7 relative h-3 bg-gray-100 rounded-sm">
            <div 
              className="absolute top-0 left-0 h-full bg-primary rounded-sm transition-all" 
              style={{ width: `${item.score * 100}%` }}
            ></div>
            {item.globalAvg > 0 && (
              <div 
                className="absolute top-[-4px] bottom-[-4px] w-0.5 bg-gray-900 z-10"
                style={{ left: `${item.globalAvg * 100}%` }}
                title={`Average: ${(item.globalAvg * 100).toFixed(1)}`}
              ></div>
            )}
          </div>
          <div className="col-span-1 text-right font-mono text-xs font-semibold text-gray-900">
            {(item.score * 100).toFixed(1)}
          </div>
        </div>
      ))}
      <div className="pt-4 mt-2 border-t flex justify-between text-xs text-gray-500">
        <div>0</div>
        <div className="flex items-center gap-1">
          <div className="w-0.5 h-2 bg-gray-900"></div>
          <span>Index Average</span>
        </div>
        <div>100</div>
      </div>
    </div>
  );
}
