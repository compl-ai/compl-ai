export class RateLimiter {
    private concurrency: number;
    private running: number = 0;
    private queue: (() => void)[] = [];

    constructor(concurrency: number) {
        this.concurrency = concurrency;
    }

    /**
     * Executes a promise-returning function with concurrency control and backoff.
     */
    async run<T>(fn: () => Promise<T>): Promise<T> {
        if (this.running >= this.concurrency) {
            await new Promise<void>(resolve => this.queue.push(resolve));
        }
        this.running++;
        try {
            return await this.withBackoff(fn);
        } finally {
            this.running--;
            if (this.queue.length > 0) {
                const next = this.queue.shift();
                next?.();
            }
        }
    }

    /**
     * Retries the function with exponential backoff if it hits 429 or 503 errors.
     */
    private async withBackoff<T>(fn: () => Promise<T>, retries = 5, delayMs = 4000): Promise<T> {
        try {
            return await fn();
        } catch (error: any) {
            // Check if it's a rate limit (429) or high demand (503) error
            const isRateLimit = error?.status === 429 || error?.code === 429 || error?.message?.includes('429');
            const isUnavailable = error?.status === 503 || error?.code === 503 || error?.message?.includes('503');
            
            if ((isRateLimit || isUnavailable) && retries > 0) {
                console.warn(`\n[RateLimiter] Hit ${isRateLimit ? '429 (Rate Limit)' : '503 (High Demand)'}. Retrying in ${delayMs / 1000}s... (${retries} retries left)`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
                return this.withBackoff(fn, retries - 1, delayMs * 2);
            }
            throw error; // If out of retries or it's a different error (like 400), throw it.
        }
    }
}
