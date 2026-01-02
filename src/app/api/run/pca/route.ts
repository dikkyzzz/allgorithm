import { NextRequest, NextResponse } from 'next/server';
import { PCA } from 'ml-pca';
import { getDatasetById } from '@/lib/datasets';
import { extractFeatures, normalizeData } from '@/lib/ml-utils';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { datasetId, nComponents = 2 } = body;

        // Get dataset
        const dataset = getDatasetById(datasetId);
        if (!dataset) {
            return NextResponse.json(
                { success: false, error: 'Dataset not found' },
                { status: 404 }
            );
        }

        // Extract and normalize features
        const features = extractFeatures(dataset.data, dataset.features);
        const normalizedFeatures = normalizeData(features);

        // Run PCA
        const pca = new PCA(normalizedFeatures, { center: true, scale: true });

        // Get results
        const explainedVariance = pca.getExplainedVariance();
        const cumulativeVariance = pca.getCumulativeVariance();
        const loadings = pca.getLoadings();
        const transformed = pca.predict(normalizedFeatures, { nComponents });

        // Calculate component contributions
        const componentContributions = dataset.features.map((feature, i) => ({
            feature,
            contributions: loadings.data.slice(0, nComponents).map((loading: number[]) =>
                Number(loading[i].toFixed(4))
            ),
        }));

        return NextResponse.json({
            success: true,
            data: {
                algorithm: 'Principal Component Analysis (PCA)',
                parameters: { nComponents },
                results: {
                    explainedVariance: explainedVariance.slice(0, nComponents).map((v: number) =>
                        Number((v * 100).toFixed(2))
                    ),
                    cumulativeVariance: cumulativeVariance.slice(0, nComponents).map((v: number) =>
                        Number((v * 100).toFixed(2))
                    ),
                    componentContributions,
                    transformedDataSample: transformed.data.slice(0, 10).map((row: number[]) =>
                        row.map(v => Number(v.toFixed(4)))
                    ),
                    totalVarianceExplained: Number((cumulativeVariance[nComponents - 1] * 100).toFixed(2)),
                },
                datasetInfo: {
                    name: dataset.name,
                    rows: dataset.rows,
                    features: dataset.features,
                },
            },
        });
    } catch (error) {
        console.error('PCA error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to run PCA analysis' },
            { status: 500 }
        );
    }
}
