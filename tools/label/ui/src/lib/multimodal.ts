export interface ContentBlock {
    type: string;
    text?: string;
    image?: string;
    detail?: string;
}

export interface Message {
    role: string;
    content: ContentBlock[] | string;
}

export function parseMultimodalInput(input: any): Message[] {
    if (!input) return [];
    
    // If it's already an array of messages (like MMMU Pro)
    if (Array.isArray(input)) {
        return input;
    }
    
    // If it's a plain string (like BBQ or INCLUDE)
    if (typeof input === 'string') {
        return [{ role: 'user', content: input }];
    }
    
    // Fallback for weird objects
    return [{ role: 'user', content: JSON.stringify(input) }];
}

// Formats for Gemini API (returns an array of 'Part' objects)
export function formatGeminiParts(input: any): any[] {
    const messages = parseMultimodalInput(input);
    const parts: any[] = [];
    
    for (const msg of messages) {
        if (typeof msg.content === 'string') {
            parts.push({ text: msg.content });
        } else if (Array.isArray(msg.content)) {
            for (const block of msg.content) {
                if (block.type === 'text' && block.text) {
                    parts.push({ text: block.text });
                } else if (block.type === 'image' && block.image) {
                    // Extract mimeType and base64 data from data URI
                    const match = block.image.match(/^data:(image\/[a-zA-Z+.-]+);base64,(.+)$/);
                    if (match) {
                        parts.push({
                            inlineData: {
                                mimeType: match[1],
                                data: match[2]
                            }
                        });
                    } else {
                        // Fallback if it's not a valid data URI
                        parts.push({ text: "[Image unparseable]" });
                    }
                }
            }
        }
    }
    return parts;
}
