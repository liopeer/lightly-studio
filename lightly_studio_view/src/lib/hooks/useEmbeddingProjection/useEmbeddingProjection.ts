import { projectEmbedding } from '$lib/api/lightly_studio_local/sdk.gen';
import type { TextEmbedding } from '$lib/hooks/useGlobalStorage';
import { createQuery } from '@tanstack/svelte-query';
import { derived, type Readable } from 'svelte/store';

export type ProjectedPoint = { x: number; y: number };

export function useEmbeddingProjection(
    collectionId: string,
    embedding: Readable<TextEmbedding | undefined>
) {
    return createQuery(
        derived(embedding, ($embedding) => ({
            queryKey: ['projection', collectionId, $embedding?.embedding],
            queryFn: async (): Promise<ProjectedPoint> => {
                const { data, error } = await projectEmbedding({
                    path: { collection_id: collectionId },
                    body: { embedding: $embedding!.embedding },
                    throwOnError: false
                });
                if (error || !data) {
                    throw new Error('Projection failed');
                }
                return { x: data.x, y: data.y };
            },
            enabled: !!$embedding,
            retry: false
        }))
    );
}
