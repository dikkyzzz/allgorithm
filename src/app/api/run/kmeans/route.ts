import { NextRequest, NextResponse } from 'next/server';
import { kmeans } from 'ml-kmeans';
import { getDatasetById } from '@/lib/datasets';
import { extractFeatures, normalizeData, silhouetteScore, calculateInertia } from '@/lib/ml-utils';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { datasetId, k = 3, maxIterations = 100, initialization = 'kmeans++' } = body;

        // Get dataset
        const dataset = getDatasetById(datasetId);
        if (!dataset) {
            return NextResponse.json(
                { success: false, error: 'Dataset not found' },
                { status: 404 }
            );
        }

        // Extract numeric features
        const features = extractFeatures(dataset.data, dataset.features);
        const normalizedFeatures = normalizeData(features);

        // Run K-Means
        const result = kmeans(normalizedFeatures, k, {
            initialization: initialization as 'random' | 'kmeans++',
            maxIterations,
        });

        // Calculate metrics
        const labels = result.clusters;
        const centroids = result.centroids;
        const inertia = calculateInertia(normalizedFeatures, labels, centroids);
        const silhouette = silhouetteScore(normalizedFeatures, labels);

        // Calculate cluster sizes
        const clusterSizes = labels.reduce((acc: Record<number, number>, label) => {
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, {});

        return NextResponse.json({
            success: true,
            data: {
                algorithm: 'K-Means',
                parameters: { k, maxIterations, initialization },
                results: {
                    clusters: labels,
                    centroids: centroids,
                    inertia: Number(inertia.toFixed(4)),
                    silhouetteScore: Number(silhouette.toFixed(4)),
                    clusterSizes,
                    iterations: result.iterations,
                    converged: result.converged,
                },
                datasetInfo: {
                    name: dataset.name,
                    rows: dataset.rows,
                    features: dataset.features,
                },
            },
        });
    } catch (error) {
        console.error('K-Means error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to run K-Means clustering' },
            { status: 500 }
        );
    }
}
