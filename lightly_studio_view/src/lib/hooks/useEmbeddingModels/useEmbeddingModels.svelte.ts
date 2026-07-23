import { readEmbeddingModelsOptions } from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { createQuery } from '@tanstack/svelte-query';

export function useEmbeddingModels(getParams: () => { collectionId: string }) {
    return createQuery(() =>
        readEmbeddingModelsOptions({
            path: { collection_id: getParams().collectionId }
        })
    );
}
